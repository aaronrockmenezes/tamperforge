#!/usr/bin/env python
"""Does Gemma's post-block RMSNorm leak an ablated direction back into the residual stream?

THE MECHANISM. Gemma-3 has FOUR norms per decoder layer -- input_layernorm,
post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm -- where Qwen3
and Llama-3 have two. The extra pair normalises the ATTENTION OUTPUT and the MLP OUTPUT before
they are added to the residual stream. Qwen/Llama add raw block outputs.

Weight-space abliteration does W_out <- W_out - d d^T W_out, so the block output o satisfies
<o, d> = 0 exactly. On a pre-norm-only model that lands in the residual untouched and the
direction is gone. On Gemma it first passes through RMSNorm:

    y = diag(gamma) * (o / rms(o))
    <y, d>  =  <diag(gamma) o, d> / rms(o)  =  <o, diag(gamma) d> / rms(o)

o is orthogonal to d, but o is NOT generally orthogonal to diag(gamma)*d. So the learned
per-dimension gain can rotate the ablated output back onto the direction that was just removed.
If gamma were constant the two would be parallel and the leak would be exactly zero -- the leak
is driven entirely by the VARIANCE of gamma.

METRIC (pure weight space, no forward pass):
    leak_angle = sin(angle between d and diag(gamma) d)
               = || diag(gamma)d - proj_d(diag(gamma)d) ||  /  || diag(gamma)d ||
    0.0 => gamma acts as a pure scalar on d; ablation survives the norm intact.
    high => the norm mixes other dimensions back onto d; ablation is partially undone.

An expected-leak estimate is also reported: for a random o drawn orthogonal to d, the expected
|<o, diag(gamma)d>| / (||o|| ||diag(gamma)d||), which is what actually reaches the residual.

Reports every norm in every layer, so Gemma's post_attention/post_feedforward norms can be
compared against its own input/pre_feedforward norms (which sit BEFORE the block and cannot
leak this way) and against Qwen/Llama, which have no post-block norms at all.

  python scripts/probes/postnorm_leak_test.py --model-id google/gemma-3-1b-it --layer 14
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

from tamperforge import capture_residuals, load_model  # noqa: E402
from tamperforge.data import load_advbench  # noqa: E402
from tamperforge.data_p1b import load_benign_instructions  # noqa: E402

# Norms that sit AFTER a block and therefore rescale what is written to the residual stream.
# Gemma-3 only. Everything else in the list is a pre-block norm and is measured for contrast.
POST_BLOCK = {"post_attention_layernorm", "post_feedforward_layernorm"}


def leak_metrics(gamma: torch.Tensor, d: torch.Tensor) -> dict:
    """How far diag(gamma)d rotates off d, and the expected leak for o drawn orthogonal to d."""
    g = gamma.double()
    dd = d.double()
    dd = dd / dd.norm()
    gd = g * dd                                  # diag(gamma) d
    if gd.norm() < 1e-12:
        return {"leak_angle_sin": 0.0, "expected_leak": 0.0, "gamma_cv": 0.0}
    par = torch.dot(gd, dd) * dd                 # component parallel to d
    perp = gd - par
    leak_sin = float(perp.norm() / gd.norm())

    # Expected |<o, gd>| / (||o|| ||gd||) for o uniform on the sphere orthogonal to d.
    # The orthogonal complement carries ||perp||^2 of gd's energy spread over (n-1) dims, so
    # E[cos^2] = ||perp||^2 / ((n-1) ||gd||^2). Report the RMS cosine.
    n = dd.numel()
    exp_leak = float((perp.norm() / gd.norm()) / max(n - 1, 1) ** 0.5)

    # Coefficient of variation of gamma: 0 => constant gain => provably zero leak.
    # ponytail: pooled CV over the whole gain tensor. Disagrees with
    # gamma_surgical_amplification.py, which averages PER-LAYER CV: 0.88/0.77 here vs
    # 0.51/0.46 there, same tensors. ceiling: the two numbers are not comparable and must
    # never be quoted side by side. upgrade: pick one estimator and restate both results
    # in it before either appears in a write-up.
    cv = float(g.std() / g.abs().mean().clamp(min=1e-12))
    return {"leak_angle_sin": round(leak_sin, 6), "expected_leak": round(exp_leak, 8),
            "gamma_cv": round(cv, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--layer", type=int, required=True, help="layer the refusal direction is read from")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", default="results/postnorm_leak.json")
    args = ap.parse_args()

    os.environ.setdefault("TF_ATTN_IMPL", "eager")
    print(f"[load] {args.model_id}")
    model, tok, device = load_model(args.model_id)
    model = model.float().eval()

    harmful = [p for p, _ in load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv")][:args.n]
    benign = load_benign_instructions(args.n, seed=42)[:args.n]
    Hh = capture_residuals(model, tok, harmful, [args.layer], device)[args.layer].float()
    Hb = capture_residuals(model, tok, benign, [args.layer], device)[args.layer].float()
    d = Hh.mean(0) - Hb.mean(0)
    d = (d / d.norm()).cpu()
    print(f"[dir] refusal direction from layer {args.layer}, dim {d.numel()}")

    norm_names = [n for n, _ in model.model.layers[0].named_children() if "layernorm" in n]
    print(f"[arch] norms per layer: {norm_names}")
    has_post = [n for n in norm_names if n in POST_BLOCK]
    if not has_post:
        print("[arch] NO post-block norms -- this leak channel does not exist for this model.")

    rows = []
    for i, layer in enumerate(model.model.layers):
        for name in norm_names:
            mod = getattr(layer, name, None)
            if mod is None or not hasattr(mod, "weight"):
                continue
            g = mod.weight.data.detach().cpu().float()
            # Gemma RMSNorm stores (gain - 1); the effective multiplier is 1 + weight.
            if "gemma" in args.model_id.lower():
                g = g + 1.0
            m = leak_metrics(g, d)
            m |= {"layer": i, "norm": name, "post_block": name in POST_BLOCK}
            rows.append(m)

    print(f"\n{'norm':30s} {'post?':>6s} {'leak_sin':>10s} {'exp_leak':>11s} {'gamma_cv':>9s}")
    for name in norm_names:
        sub = [r for r in rows if r["norm"] == name]
        if not sub:
            continue
        avg_sin = sum(r["leak_angle_sin"] for r in sub) / len(sub)
        avg_exp = sum(r["expected_leak"] for r in sub) / len(sub)
        avg_cv = sum(r["gamma_cv"] for r in sub) / len(sub)
        print(f"{name:30s} {str(name in POST_BLOCK):>6s} {avg_sin:>10.4f} {avg_exp:>11.6f} "
              f"{avg_cv:>9.4f}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(p)) if p.exists() else {}
    prev[args.model_id] = {"layer": args.layer, "norms": norm_names,
                           "has_post_block_norms": bool(has_post), "per_norm": rows}
    json.dump(prev, open(p, "w"), indent=2)
    print(f"[saved] {p}")
    print("\n[read] leak_sin near 0 => the norm scales d without mixing; ablation survives it. "
          "leak_sin high on a POST-BLOCK norm => the learned gain rotates the ablated output "
          "back onto the removed direction, partially undoing weight-space abliteration.")


if __name__ == "__main__":
    main()
