#!/usr/bin/env python
"""Phase 0a: did representation rerouting actually CONVERGE on a given checkpoint?

WHY THIS EXISTS. version_G-Qwen's L_rr fell 0.9967 -> 0.1702 over 500 steps: attacked and
frozen-base representations went from near-identical to near-orthogonal on harmful text, which
is the mechanism working. **Gemma's L_rr has never been observed.** The gemma training runs'
`results/tamper_resistant_p1b_*/events.jsonl` were not preserved in the teardown backup (the tar
has 73 events.jsonl files, every one of them from JUDGING, none from training), and the periodic
step line does not print L_rr -- it only ever lived in that events file.

So the entire "gemma is architecturally different" investigation rests on an unverified premise:
that rerouting trained at all on gemma. If L_rr never dropped, gemma's defence was produced by
lambda_safe/lambda_uncensor alone -- which ARE inherently direction-specific -- and that alone
explains the observed rank-1-fires / surgical-evades asymmetry, with no architectural story
needed.

Corroborating hint from the surviving step logs: gemma version_G's refuse clean->ablated gap
oscillates violently (1.90, 11.38, 7.44, 11.13, ...) and then COLLAPSES at the end -- step 450
is 1.47->1.54 and step 500 is 1.33->3.11. The rr8 arm ends at 1.36->1.46 and 1.32->1.40, i.e. no
gap at all. Qwen's held a 2-5x gap throughout.

WHAT THIS DOES. Reproduces `_reroute_loss` (train_tamper_resistant_v8.py:577) exactly --
teacher-force the ABLATED model and the FROZEN BASE on the same harmful prompt+completion,
mean relu(cosine) between their hidden states over the rr layers -- against a trained checkpoint,
with no training. Also reports the same quantity for the UNTRAINED base as a ceiling, since
L_rr ~ 1.0 is what "no rerouting whatsoever" looks like.

Reading it:
  gemma L_rr ~ Qwen's (~0.2)  => rerouting converged; the failure is generalisation, go Phase 1B.
  gemma L_rr near the base ceiling (~1.0) => it never trained; go Phase 1A (log L_rr, sweep
                                            lambda_rr, extend steps) and drop the architecture
                                            theories.

  python scripts/probes/posthoc_lrr.py --model-id Qwen/Qwen3-0.6B \
      --checkpoint /tmp/vg_ckpt/.../version_g_qwen_500.pt --layer 20 --tag qwen_vg
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

READ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj")
WRITE = ("self_attn.o_proj", "mlp.down_proj")


def _empty_cache():
    """`del` alone does not return MPS/CUDA blocks to the allocator."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_trained(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt.pop("_meta", None)
    named = dict(model.named_parameters())
    n = 0
    for name, t in ckpt.items():
        if name in named:
            named[name].data.copy_(t.to(named[name].dtype).to(named[name].device))
            n += 1
    if n == 0:
        sys.exit(f"FATAL: 0 compatible matrices from {ckpt_path}")
    print(f"[ckpt] loaded {n} matrices")
    return n


def ablated_overrides(model, d, layers):
    """{param_name: ablated_weight} -- the canonical rank-1 attack, as the trainer builds it."""
    params = dict(model.named_parameters())
    dev = next(iter(params.values())).device
    d = (d / d.norm()).to(dev)
    ov = {}
    for li in layers:
        base = f"model.layers.{li}."
        for name in READ:
            k = base + name + ".weight"
            ov[k] = project_out_read(params[k].data.float(), d).to(params[k].dtype)
        for name in WRITE:
            k = base + name + ".weight"
            ov[k] = project_out_write(params[k].data.float(), d).to(params[k].dtype)
    return ov


@torch.no_grad()
def reroute_loss(model, tok, pairs, device, W0, overrides, rr_layers, max_len=320,
                 debug=False, center=False):
    """Verbatim reproduction of _reroute_loss, minus the gradient.

    The debug block exists because the first gemma run returned exactly 0.0000 while Qwen's
    same-arm control returned 0.9854. 0.0000 is what the degenerate paths produce (cnt==0, or
    an all-zero / non-finite `a` making cosine_similarity return 0 via its eps-clamped
    denominator) -- it is NOT a low-similarity reading. Anything that can silently yield it
    now raises instead.
    """
    total, cnt = 0.0, 0
    for prompt, completion in pairs:
        prefix = apply_chat_template_no_think(
            tok, [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        enc = tok(prefix + completion, return_tensors="pt", truncation=True,
                  max_length=max_len).to(device)
        plen = min(tok(prefix, return_tensors="pt")["input_ids"].shape[1],
                   enc["input_ids"].shape[1] - 1)
        h_att = functional_call(model, overrides, args=(),
                                kwargs={**enc, "output_hidden_states": True}).hidden_states
        h_ref = functional_call(model, W0, args=(),
                                kwargs={**enc, "output_hidden_states": True}).hidden_states
        for li in rr_layers:
            a = h_att[li][:, plen:, :].float()
            b = h_ref[li][:, plen:, :].float()
            if a.shape[1] == 0:
                continue
            if debug and cnt < len(rr_layers):
                print(f"    [dbg] L{li:2d} tok={a.shape[1]:3d} "
                      f"|a|={float(a.norm()):9.3f} |b|={float(b.norm()):9.3f} "
                      f"finite_a={bool(a.isfinite().all())} finite_b={bool(b.isfinite().all())}")
            if not (a.isfinite().all() and b.isfinite().all()):
                raise RuntimeError(f"non-finite hidden states at layer {li} "
                                   f"(a_finite={bool(a.isfinite().all())}, "
                                   f"b_finite={bool(b.isfinite().all())})")
            if float(a.norm()) < 1e-6 or float(b.norm()) < 1e-6:
                raise RuntimeError(f"degenerate zero hidden state at layer {li}: "
                                   f"|a|={float(a.norm()):.3e} |b|={float(b.norm()):.3e}")
            if center:
                mu = b.mean(dim=1, keepdim=True)
                a, b = a - mu, b - mu
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            total += float(torch.relu(cos).mean())
            cnt += 1
    if cnt == 0:
        raise RuntimeError("cnt==0: every pair had zero completion tokens after plen; "
                           "L_rr is undefined, not 0.0")
    return total / cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--layer", type=int, required=True, help="direction layer")
    ap.add_argument("--harm-targets", default="data/harm_targets_qwen.json")
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--n-direction", type=int, default=128)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/posthoc_lrr.json")
    ap.add_argument("--arms", default="base_ceiling,trained",
                    help="comma list; run just base_ceiling first to validate cheaply")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--center", action="store_true",
                    help="match --rr-center: subtract the frozen base's per-position mean from "
                         "both streams. Answers whether a checkpoint rerouted CONTENT while the "
                         "DC component hid it from the uncentred metric.")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    os.environ.setdefault("TF_ATTN_IMPL", "eager")
    print(f"[load] {args.model_id}")
    model, tok, device = load_model(args.model_id)
    n_layers = len(model.model.layers)
    # +1 because hidden_states is [emb, layer0_out, ...], so layer i lives at index i+1. The
    # trainer does this (train_tamper_resistant_v8.py:1377) and the first version of this probe
    # did not, so it compared indices 13..25 where training optimised 14..26. Adjacent layers,
    # so it did not change any verdict, but it is a real off-by-one against the reference.
    rr_layers = [li + 1 for li in range(n_layers // 2, n_layers)]   # --rr-layers last_half
    # Free it IMMEDIATELY. This model is only ever read for n_layers/tok/device, but keeping it
    # alive costs a full fp32 copy for the whole run: with W0_base's clones, `m`, and `ov` that
    # is ~15GB for gemma-1b on a 16GB unified machine, and MPS answers an allocation it cannot
    # serve with SILENT ZEROS rather than an OOM. That is what produced gemma's "L_rr = 0.0000"
    # -- the frozen-base pass returned an all-zero hidden state. Qwen3-0.6B is small enough that
    # four copies fit, which is why the harness looked validated.
    del model
    _empty_cache()
    print(f"[cfg] {n_layers} layers, rr_layers = last_half = {rr_layers[0]}..{rr_layers[-1]}")

    mined = json.loads((ROOT / args.harm_targets).read_text())
    pairs = [(g, c[0]) for g, c in list(mined.items()) if c][:args.n_pairs]
    print(f"[data] {len(pairs)} (harmful prompt, real harmful completion) pairs")

    benign = load_benign_instructions(args.n_direction, seed=42)
    harmful = [p for p, _ in load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv")][:args.n_direction]

    # CRITICAL, and the thing I first got wrong: in training, W0 is snapshotted at step 0, i.e.
    # it is the ORIGINAL BASE weights (train_tamper_resistant_v8.py:1275, inside the loop that
    # flips requires_grad on). So _reroute_loss compares the ABLATED CURRENT model against the
    # FROZEN BASE -- not against itself. Measuring cos(ablated_trained, trained) instead gives a
    # completely different number (Qwen read 0.8194 that way vs its true training value 0.1702).
    base_model, _, _ = load_model(args.model_id)
    base_model = base_model.float().eval()
    W0_base = {k: v.detach().clone() for k, v in base_model.named_parameters()}
    del base_model
    _empty_cache()
    print(f"[W0] frozen base snapshot: {len(W0_base)} params (this is what training compares to)")

    out = {}
    for label, ckpt in (("base_ceiling", None), ("trained", args.checkpoint)):
        if label not in arms:
            continue
        m, _, _ = load_model(args.model_id)
        m = m.float().eval()
        if ckpt:
            load_trained(m, ckpt)
        # Direction is recomputed on the CURRENT model, as the trainer does every 25 steps.
        Hh = capture_residuals(m, tok, harmful, [args.layer], device)[args.layer].float()
        Hb = capture_residuals(m, tok, benign, [args.layer], device)[args.layer].float()
        d = Hh.mean(0) - Hb.mean(0)
        d = d / d.norm()

        if args.debug:
            print(f"  [dbg] {label}: |d|={float(d.norm()):.4f} "
                  f"finite_d={bool(d.isfinite().all())} "
                  f"d_absmax={float(d.abs().max()):.4f}")
        ov = ablated_overrides(m, d, list(range(n_layers)))       # ablated CURRENT weights
        val = reroute_loss(m, tok, pairs, device, W0_base, ov, rr_layers, debug=args.debug,
                           center=args.center)
        out[label] = round(val, 4)
        print(f"  {label:14s} L_rr = {val:.4f}", flush=True)
        del m, ov, Hh, Hb
        _empty_cache()

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(p)) if p.exists() else {}
    prev[args.tag] = {"model_id": args.model_id, "checkpoint": args.checkpoint,
                      "layer": args.layer, "n_pairs": len(pairs),
                      "rr_layers": [rr_layers[0], rr_layers[-1]],
                      "centered": args.center, **out}
    json.dump(prev, open(p, "w"), indent=2)
    print(f"[saved] {p} (tag {args.tag})")
    print("[ref] version_G-Qwen's TRAINING L_rr ended at 0.1702 (step 500), started 0.9967")


if __name__ == "__main__":
    main()
