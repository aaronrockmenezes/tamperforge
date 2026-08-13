#!/usr/bin/env python
"""How entangled is refusal with capability? Cross-architecture, on BASE models. Local/MPS.

THE QUESTION. TamperForge's whole thesis is the poison pill: entangle safety with capability so
that removing safety costs capability. That is only possible if the refusal direction ALREADY
lives partly inside the capability subspace -- if refusal is cleanly separable, an attacker can
delete it without touching anything you care about, and no training objective can change the
underlying geometry.

version_G reproduced on Qwen (all 3 gates) and Llama (partial), and failed on Gemma across four
separate objectives (G/H/I/J). The Gemma signature: surgical k16 reaches 0.94 harm while rank-1
reaches 0.11 -- i.e. once you project the capability-overlapping component OUT of the refusal
direction, what remains still fully mediates refusal. That is what low entanglement looks like.

This measures it directly, so "can the poison pill work on this architecture at all?" becomes a
screening test you run BEFORE spending a training run.

TWO METRICS.
  rank1_overlap  = ||P_cap d|| / ||d||  for the rank-1 mean-diff direction.
      The repo's existing metric (experiments/v11_training_direction_overlap.py). Measured 0.69
      for Qwen3-0.6B at layer 20 / cap-rank 4.
  subspace_overlap = mean(sigma_i^2), sigma = singular values of U_refusal^T U_cap.
      Mean squared cosine of principal angles between the two subspaces -- the standard
      subspace-similarity measure (cf. arXiv:2606.14388, arXiv:2505.14185). Generalises the
      rank-1 number to the multi-dimensional case, which matters because refusal is NOT rank-1:
      it is mediated by multiple independent directions forming concept cones
      (arXiv:2502.17420, ICML 2025).

WHY BOTH. rank1_overlap answers "is the direction we train against entangled?". subspace_overlap
answers "is the refusal MECHANISM entangled?". If rank1 is high but subspace is low, the mean-diff
direction is entangled but the cone around it is not -- an attacker just moves within the cone,
which is exactly the observed rank-1-vs-surgical gap.

ROBUST ESTIMATOR. --winsorize applies per-dimension winsorization before the mean-diff. Gemma is
reported to need winsorized steering vectors (99.5th pct) because activation outliers otherwise
dominate the direction estimate; the repo's empirical_refusal_direction does a plain mean with no
outlier handling. If Gemma's overlap changes materially under winsorization, the direction every
Gemma arm hardened was partly an outlier artifact rather than the refusal mechanism.

  python scripts/probes/refusal_capability_overlap.py --model-id google/gemma-3-1b-it
  python scripts/probes/refusal_capability_overlap.py --model-id Qwen/Qwen3-0.6B --layer 20
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


def winsorized_mean(H: torch.Tensor, pct: float) -> torch.Tensor:
    """Per-dimension winsorized mean. Clamps each feature to [pct, 1-pct] quantiles first."""
    lo = torch.quantile(H, pct, dim=0, keepdim=True)
    hi = torch.quantile(H, 1.0 - pct, dim=0, keepdim=True)
    return H.clamp(min=lo, max=hi).mean(0)


@torch.no_grad()
def cap_subspace(model, tok, device, prompts, layer, rank):
    """Top-`rank` right singular directions of centred capability activations. [rank, d_model]"""
    H = capture_residuals(model, tok, prompts, [layer], device)[layer].float()
    H = H - H.mean(0, keepdim=True)
    _, s, Vh = torch.linalg.svd(H, full_matrices=False)
    var = float(s[:rank].pow(2).sum() / s.pow(2).sum().clamp(min=1e-9))
    return Vh[:rank], var


@torch.no_grad()
def refusal_dirs(model, tok, device, harmful, benign, layer, rank, winsor):
    """(rank-1 mean-diff direction, top-`rank` SVD refusal subspace)."""
    Hh = capture_residuals(model, tok, harmful, [layer], device)[layer].float()
    Hb = capture_residuals(model, tok, benign, [layer], device)[layer].float()
    if winsor > 0:
        d1 = winsorized_mean(Hh, winsor) - winsorized_mean(Hb, winsor)
    else:
        d1 = Hh.mean(0) - Hb.mean(0)
    d1 = d1 / d1.norm().clamp(min=1e-8)
    # Refusal SUBSPACE: SVD of (harmful - mean benign), the rank_k_svd attack's estimator.
    D = Hh - Hb.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    return d1, Vh[:rank]


def principal_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean squared cosine of principal angles between rowspace(A) and rowspace(B).

    Both are orthonormalised first -- SVD right-singular rows already are, but the mean-diff
    direction is not part of an orthonormal set, so do it explicitly and be safe.
    """
    Qa = torch.linalg.qr(A.T)[0]          # [d, ra]
    Qb = torch.linalg.qr(B.T)[0]          # [d, rb]
    s = torch.linalg.svdvals(Qa.T @ Qb)   # cosines of principal angles
    return float(s.pow(2).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--layer", type=int, default=None,
                    help="default: the repo's known-good direction layer for this arch")
    ap.add_argument("--cap-ranks", default="4,16")
    ap.add_argument("--refusal-rank", type=int, default=8)
    ap.add_argument("--n", type=int, default=128, help="prompts per side")
    ap.add_argument("--winsorize", type=float, default=0.0,
                    help="per-dim winsorization fraction, e.g. 0.005 for 99.5th pct")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default="results/refusal_capability_overlap.json")
    args = ap.parse_args()

    # Direction layers established in this repo, per architecture. CLAUDE.md: never scale these
    # between architectures -- these are measured, not proportional guesses.
    DEFAULT_LAYER = {"qwen": 20, "llama": 13, "gemma": 14}
    if args.layer is None:
        key = next((k for k in DEFAULT_LAYER if k in args.model_id.lower()), None)
        if key is None:
            sys.exit("--layer required for an architecture with no recorded default")
        args.layer = DEFAULT_LAYER[key]

    os.environ.setdefault("TF_ATTN_IMPL", "eager")
    print(f"[load] {args.model_id} layer={args.layer} winsorize={args.winsorize}")
    model, tok, device = load_model(args.model_id)
    if args.dtype == "float32":
        model = model.float()
    model.eval()

    harmful = [p for p, _ in load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv")][:args.n]
    benign = load_benign_instructions(args.n, seed=42)[:args.n]
    # Capability prompts must be a DIFFERENT distribution from the benign refusal contrast, or
    # the "capability subspace" is just the benign half of the refusal direction and the overlap
    # is guaranteed high by construction.
    cap = load_benign_instructions(args.n * 3, seed=7)[args.n:args.n + args.n]
    print(f"[data] harmful {len(harmful)} | benign {len(benign)} | capability {len(cap)}")

    d1, R = refusal_dirs(model, tok, device, harmful, benign, args.layer,
                         args.refusal_rank, args.winsorize)

    out = {"model_id": args.model_id, "layer": args.layer, "n": args.n,
           "winsorize": args.winsorize, "refusal_rank": args.refusal_rank, "by_cap_rank": {}}
    for rank in [int(x) for x in args.cap_ranks.split(",")]:
        V, var = cap_subspace(model, tok, device, cap, args.layer, rank)
        # rank-1 metric: ||P_cap d|| / ||d||, d already unit-norm
        proj = (V.T @ (V @ d1)).norm().item()
        sub = principal_overlap(R, V)
        out["by_cap_rank"][str(rank)] = {
            "cap_variance_captured": round(var, 4),
            "rank1_overlap": round(proj, 4),
            "subspace_overlap": round(sub, 4),
        }
        print(f"  cap-rank {rank:>2d}: rank1_overlap {proj:.4f} | "
              f"subspace_overlap(refusal r{args.refusal_rank}) {sub:.4f} | "
              f"cap var captured {var:.1%}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(p)) if p.exists() else {}
    key = f"{args.model_id}|L{args.layer}|w{args.winsorize}"
    prev[key] = out
    json.dump(prev, open(p, "w"), indent=2)
    print(f"[saved] {p}  (key: {key})")


if __name__ == "__main__":
    main()
