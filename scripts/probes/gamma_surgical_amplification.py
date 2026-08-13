#!/usr/bin/env python
"""Does gemma's post-block RMSNorm gain AMPLIFY the surgical perturbation?

THE SHARPENED QUESTION THIS ATTACKS. version_G's poison pill fires on rank-1 for both Qwen and
gemma, but on surgical only for Qwen (gemma: GSM8K 0.2335, gib 0.00, harm 0.94). "Why does gemma
fail" is the wrong question; "why does the defense GENERALISE from the trained direction to a
neighbouring one on Qwen but not gemma" is the right one.

THE HYPOTHESIS. Weight-space abliteration makes <o, d> = 0 for the raw block output o. On
Qwen/Llama o lands in the residual untouched, so the residual really loses d. gemma-3 first
applies a post-block RMSNorm, y = gamma * (o / rms(o)), and

    <y, d>  proportional to  <o, gamma * d>

so what the RESIDUAL actually loses is governed by gamma*d, not d. gamma is not a mild
reweighting -- measured CV is 0.88 / 0.77 on gemma's two post-block norms, and normalize(gamma*d)
is only ~0.68 aligned with d (results/gamma_compensated_ablation/summary.json).

If a near-diagonal map with that much spread ROTATES nearby directions apart, then two attacks
that look almost identical in Euclidean terms (d and d_surgical) induce very DIFFERENT residual
edits on gemma while inducing near-identical ones on Qwen. A defense trained to fire on one would
then transfer on Qwen and not on gemma -- which is exactly the observed asymmetry.

WHAT IS MEASURED. For cap_rank k in {2,4,8,16}:

    cos_euclid   = |<d, d_surg_k>|                                  (how similar the attacks look)
    cos_gamma    = |<normalize(g*d), normalize(g*d_surg_k)>|        (how similar their residual
                                                                     edits actually are), per
                                                                     post-block norm g
    amplification = (1 - cos_gamma) / (1 - cos_euclid)

amplification ~ 1 => gamma is irrelevant, the directions stay as close as they look.
amplification >> 1 => gamma pulls them apart; generalisation failure is a norm-geometry effect.

THE CONTROL IS THE WHOLE POINT. Qwen/Llama have NO post-block norms, so for them gamma is
identically 1, cos_gamma == cos_euclid, and amplification == 1 BY CONSTRUCTION. Any Qwen row
that is not 1.000 means this script is broken, not that Qwen amplifies. Two harness bugs today
produced plausible numbers and both were caught by a control disagreeing with a known value;
this one carries its control inline.

No generation, no judge, no money -- one capture_residuals pass and some linear algebra.

  python scripts/probes/gamma_surgical_amplification.py --model-id google/gemma-3-1b-it \
      --layer 14 --tag gemma_base
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tamperforge import capture_residuals, load_model  # noqa: E402
from tamperforge.data import load_advbench  # noqa: E402
from tamperforge.data_p1b import load_benign_instructions  # noqa: E402
from tamperforge.directions import empirical_refusal_directions  # noqa: E402
from version_a_attack import _subspaces, _surgical  # noqa: E402

# gemma-3's two genuine post-block norms: these sit between a sublayer's output and the
# residual add, which is the only place a gain can rotate an already-projected direction back
# in. Qwen/Llama have no member of this set -- their same-named modules are PRE-norms.
POST_BLOCK_NORMS = ("post_attention_layernorm", "post_feedforward_layernorm")


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


def is_post_block(model, name: str) -> bool:
    """True only if `name` is applied to a sublayer OUTPUT, not to the residual going in.

    Name alone is a trap: Qwen/Llama also own a `post_attention_layernorm`, but theirs is the
    PRE-MLP norm (x <- x + MLP(post_attention_layernorm(x))). Gate on architecture, since
    getting this wrong would hand every model a fake gamma and make the control meaningless.
    """
    return name in POST_BLOCK_NORMS and "gemma" in model.config.model_type.lower()


def gamma_vectors(model) -> dict[str, torch.Tensor]:
    """{norm_name: gamma} for post-block norms only. Empty dict for pre-norm architectures."""
    out: dict[str, list[torch.Tensor]] = {}
    for layer in model.model.layers:
        for name in POST_BLOCK_NORMS:
            nm = getattr(layer, name, None)
            if nm is None or not hasattr(nm, "weight") or not is_post_block(model, name):
                continue
            # gemma RMSNorm computes x * (1 + weight), so the effective gain is weight + 1.
            out.setdefault(name, []).append(nm.weight.data.detach().float() + 1.0)
    return {k: torch.stack(v) for k, v in out.items()}          # {name: [n_layers, d_model]}


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b / (a.norm() * b.norm()).clamp(min=1e-9)).abs())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--cap-ranks", default="2,4,8,16")
    ap.add_argument("--n-direction", type=int, default=128)
    ap.add_argument("--n-cap", type=int, default=128)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/gamma_surgical_amplification.json")
    args = ap.parse_args()
    cap_ranks = tuple(int(r) for r in args.cap_ranks.split(","))

    os.environ.setdefault("TF_ATTN_IMPL", "eager")
    print(f"[load] {args.model_id} layer={args.layer}")
    model, tok, device = load_model(args.model_id)
    model = model.float().eval()
    if args.checkpoint:
        load_trained(model, args.checkpoint)

    benign = load_benign_instructions(args.n_direction, seed=42)
    harmful = [p for p, _ in load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv")][:args.n_direction]
    from v11_surgical_ablation import _cap_prompts  # noqa: PLC0415
    cap = _cap_prompts(args.n_cap)
    print(f"[data] {len(harmful)} harmful / {len(benign)} benign / {len(cap)} capability")

    d = empirical_refusal_directions(model, tok, harmful, benign, [args.layer], device)[args.layer]
    d = d.float().to(device)
    d = d / d.norm()
    Hc = capture_residuals(model, tok, cap, [args.layer], device)[args.layer].float()
    Vs = _subspaces(Hc, cap_ranks)

    G = gamma_vectors(model)
    if G:
        for name, g in G.items():
            cv = float(g.std(dim=1).mean() / g.mean(dim=1).mean().abs().clamp(min=1e-9))
            print(f"[gamma] {name}: {tuple(g.shape)} mean={float(g.mean()):.4f} "
                  f"min={float(g.min()):.4f} CV={cv:.4f}")
    else:
        print("[gamma] no post-block norms (pre-norm architecture) -- "
              "gamma == 1, so cos_gamma MUST equal cos_euclid; that is the control")

    rows = {}
    print(f"\n{'k':>3} {'cos_euclid':>11} {'norm':>26} {'cos_gamma':>10} {'amplif':>8}")
    for k in cap_ranks:
        d_s, overlap = _surgical(d, Vs[k].to(device))
        ce = cos(d, d_s)
        row = {"cos_euclid": round(ce, 4), "cap_overlap_removed": round(overlap, 4),
               "cos_gamma": {}, "amplification": {}, "cos_d_gammad": {}}
        if not G:
            # The control, stated explicitly rather than skipped: with gamma == 1 the two
            # quantities are the same number, so amplification is exactly 1.
            row["cos_gamma"]["identity"] = round(ce, 4)
            row["amplification"]["identity"] = 1.0
            print(f"{k:>3} {ce:>11.4f} {'identity (no post-norm)':>26} {ce:>10.4f} {1.0:>8.3f}")
        for name, g in G.items():
            cgs, cdg = [], []
            for gi in g:                                   # per layer
                cgs.append(cos(gi * d, gi * d_s))
                cdg.append(cos(gi * d, d))
            cg = sum(cgs) / len(cgs)
            amp = (1 - cg) / max(1 - ce, 1e-9)
            row["cos_gamma"][name] = round(cg, 4)
            row["cos_gamma"][name + "_min"] = round(min(cgs), 4)
            row["amplification"][name] = round(amp, 3)
            row["cos_d_gammad"][name] = round(sum(cdg) / len(cdg), 4)
            print(f"{k:>3} {ce:>11.4f} {name:>26} {cg:>10.4f} {amp:>8.3f}")
        rows[str(k)] = row

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(p)) if p.exists() else {}
    prev[args.tag] = {"model_id": args.model_id, "checkpoint": args.checkpoint,
                      "layer": args.layer, "model_type": model.config.model_type,
                      "has_post_block_norms": bool(G), "cap_ranks": list(cap_ranks),
                      "n_direction": args.n_direction, "n_cap": len(cap), "ranks": rows}
    json.dump(prev, open(p, "w"), indent=2)
    print(f"\n[saved] {p} (tag {args.tag})")


def _selftest() -> None:
    """cos_gamma must equal cos_euclid when gamma is constant, and diverge when it is not."""
    torch.manual_seed(0)
    a = torch.randn(64); a /= a.norm()
    b = a + 0.05 * torch.randn(64); b /= b.norm()
    flat = torch.full((64,), 3.0)                       # uniform gain = pure rescale
    assert abs(cos(flat * a, flat * b) - cos(a, b)) < 1e-5, "uniform gamma must not rotate"
    spread = torch.rand(64) * 10 + 0.01                 # high-CV gain
    assert cos(spread * a, spread * b) < cos(a, b), "high-CV gamma should pull them apart"
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
