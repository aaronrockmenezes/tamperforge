#!/usr/bin/env python
"""Is L_rr's GRADIENT starved on gemma relative to Qwen?

WHY. Phase 0a (results/posthoc_lrr.json) showed rerouting never trained on gemma: L_rr went
0.9866 -> 0.9522 in 500 steps, where Qwen went 0.9854 -> 0.2458 under IDENTICAL settings
(lambda_rr 4, 500 steps, lr 1e-5, seed 42, same harm targets, same rr-layers). Same recipe, same
strength, one converged and one did not move. That asymmetry needs a cause.

THE HYPOTHESIS. L_rr is a cosine, hence scale-invariant -- the LOSS cannot tell you how large
the residual is. Its GRADIENT can: differentiating through a normalised quantity introduces a
1/|h| factor. Gemma's residual norms are enormous (|h| ~ 9e4-2.4e5 at layers 13-25, measured in
the posthoc_lrr debug run) and its post-block RMSNorm gains average 20-35, where Qwen's residual
is orders of magnitude smaller and has no post-block gain at all. If that costs ~100x in gradient
magnitude, then lambda_rr=4 on gemma is doing the work of lambda_rr=0.04, and a {8,16,32} sweep
fails exactly like lambda_rr=4 did -- which would read, wrongly, as "rerouting does not work on
gemma".

WHY A RATIO AND NOT AN ABSOLUTE. Raw gradient norms are not comparable across architectures:
different depth, width, weight scale, and parameter count all move them. So this measures, WITHIN
each model, the ratio

    ||d L_rr / dW||  /  ||d L_lm / dW||

against a plain LM loss on benign text -- a term every model carries and whose gradient scale is
whatever "a normal training signal" means for that architecture. A ratio 100x smaller on gemma
than on Qwen says L_rr is being drowned out there and nowhere else.

SCOPE. Grads are taken only over the matrices the trainer actually updates under its default
--train-scope mlp (gate/up/down), and only over a few layers, so this fits in 16GB alongside the
functional_call double-forward. The ratio is the output; absolute magnitudes are reported only
so a degenerate zero is visible rather than silent.

  python scripts/probes/rr_gradient_scale.py --model-id Qwen/Qwen3-0.6B --layer 20 --tag qwen
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import (  # noqa: E402
    apply_chat_template_no_think, capture_residuals, load_model,
    project_out_read, project_out_write,
)
from tamperforge.data import load_advbench  # noqa: E402
from tamperforge.data_p1b import load_benign_instructions  # noqa: E402

READ = ("mlp.gate_proj", "mlp.up_proj")
WRITE = ("mlp.down_proj",)


def _empty_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def grad_norm(params) -> float:
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.detach().float().pow(2).sum())
    return tot ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--layer", type=int, required=True, help="direction layer")
    ap.add_argument("--grad-layers", type=int, default=3,
                    help="how many of the topmost layers to take grads over (memory)")
    ap.add_argument("--n-pairs", type=int, default=2)
    ap.add_argument("--n-direction", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--harm-targets", default="data/harm_targets_qwen.json")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/rr_gradient_scale.json")
    args = ap.parse_args()

    os.environ.setdefault("TF_ATTN_IMPL", "eager")
    model, tok, device = load_model(args.model_id)
    model = model.float()
    n_layers = len(model.model.layers)
    rr_layers = list(range(n_layers // 2, n_layers))
    grad_layers = list(range(n_layers - args.grad_layers, n_layers))
    print(f"[cfg] {args.model_id}: {n_layers} layers, rr_layers {rr_layers[0]}..{rr_layers[-1]}, "
          f"grads over layers {grad_layers}")

    trainable = {f"model.layers.{li}.{n}.weight" for li in grad_layers for n in READ + WRITE}
    params = []
    for n, p in model.named_parameters():
        p.requires_grad_(n in trainable)
        if n in trainable:
            params.append(p)
    print(f"[cfg] {len(params)} trainable matrices")

    benign = load_benign_instructions(args.n_direction, seed=42)
    harmful = [p for p, _ in load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv")][:args.n_direction]
    with torch.no_grad():
        Hh = capture_residuals(model, tok, harmful, [args.layer], device)[args.layer].float()
        Hb = capture_residuals(model, tok, benign, [args.layer], device)[args.layer].float()
        d = Hh.mean(0) - Hb.mean(0)
        d = (d / d.norm()).to(device)

    # Residual magnitude, the quantity the hypothesis is about. Per-element RMS, so that width
    # differences between the two models do not masquerade as a scale difference.
    with torch.no_grad():
        enc0 = tok("Explain how photosynthesis works.", return_tensors="pt").to(device)
        hs = model(**enc0, output_hidden_states=True).hidden_states
        rms = {li: float(hs[li].float().pow(2).mean().sqrt()) for li in rr_layers}
    print(f"[resid] per-element RMS over rr_layers: min={min(rms.values()):.2f} "
          f"mean={sum(rms.values())/len(rms):.2f} max={max(rms.values()):.2f}")

    mined = json.loads((ROOT / args.harm_targets).read_text())
    pairs = [(g, c[0]) for g, c in list(mined.items()) if c][:args.n_pairs]

    named = dict(model.named_parameters())
    W0 = {k: v.detach().clone() for k, v in named.items()}

    def overrides():
        """Differentiable ablated weights, as _ablated_overrides builds them."""
        ov = {}
        for li in range(n_layers):
            base = f"model.layers.{li}."
            for n in READ:
                k = base + n + ".weight"
                ov[k] = project_out_read(named[k].float(), d).to(named[k].dtype)
            for n in WRITE:
                k = base + n + ".weight"
                ov[k] = project_out_write(named[k].float(), d).to(named[k].dtype)
        return ov

    # --- L_rr ---
    model.zero_grad(set_to_none=True)
    total, cnt = torch.zeros((), device=device), 0
    ov = overrides()
    for prompt, completion in pairs:
        prefix = apply_chat_template_no_think(
            tok, [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = tok(prefix + completion, return_tensors="pt", truncation=True,
                  max_length=args.max_len).to(device)
        plen = min(tok(prefix, return_tensors="pt")["input_ids"].shape[1],
                   enc["input_ids"].shape[1] - 1)
        h_att = functional_call(model, ov, args=(),
                                kwargs={**enc, "output_hidden_states": True}).hidden_states
        with torch.no_grad():
            h_ref = functional_call(model, W0, args=(),
                                    kwargs={**enc, "output_hidden_states": True}).hidden_states
        for li in rr_layers:
            a, b = h_att[li][:, plen:, :].float(), h_ref[li][:, plen:, :].float()
            if a.shape[1] == 0:
                continue
            if float(b.norm()) < 1e-6:
                sys.exit(f"FATAL: zero reference hidden state at layer {li} -- almost certainly "
                         f"an MPS silent-zero under memory pressure, not a real value. "
                         f"Lower --grad-layers / --n-pairs.")
            total = total + torch.relu(torch.nn.functional.cosine_similarity(a, b, dim=-1)).mean()
            cnt += 1
    L_rr = total / max(cnt, 1)
    L_rr.backward()
    g_rr = grad_norm(params)
    print(f"  L_rr    = {float(L_rr):.4f}   ||grad|| = {g_rr:.6e}")
    del ov, total
    _empty_cache()

    # --- L_lm on benign text, the within-model scale reference ---
    model.zero_grad(set_to_none=True)
    texts = [apply_chat_template_no_think(tok, [{"role": "user", "content": p}],
                                          tokenize=False, add_generation_prompt=True) + " Sure."
             for p in benign[:args.n_pairs]]
    lm = torch.zeros((), device=device)
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
        lm = lm + model(**enc, labels=enc["input_ids"]).loss
    L_lm = lm / len(texts)
    L_lm.backward()
    g_lm = grad_norm(params)
    print(f"  L_lm    = {float(L_lm):.4f}   ||grad|| = {g_lm:.6e}")

    ratio = g_rr / max(g_lm, 1e-30)
    print(f"  ratio ||dL_rr|| / ||dL_lm|| = {ratio:.6e}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(p)) if p.exists() else {}
    prev[args.tag] = {"model_id": args.model_id, "layer": args.layer, "n_layers": n_layers,
                      "grad_layers": grad_layers, "n_pairs": len(pairs),
                      "resid_rms_mean": round(sum(rms.values()) / len(rms), 3),
                      "resid_rms_max": round(max(rms.values()), 3),
                      "L_rr": round(float(L_rr), 4), "grad_rr": g_rr,
                      "L_lm": round(float(L_lm), 4), "grad_lm": g_lm,
                      "ratio_rr_over_lm": ratio}
    json.dump(prev, open(p, "w"), indent=2)
    print(f"[saved] {p} (tag {args.tag})")


if __name__ == "__main__":
    main()
