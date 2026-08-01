#!/usr/bin/env python
"""Export a checkpoint ablated at a chosen ablation STRENGTH, to find the collapse trigger.

Why this exists: real Heretic beats version_A with a LOWER-perturbation attack than the one
it beats it with less well (t175, KL 0.0124 -> 0.2615 harmful, vs t85, KL 0.0278 -> 0.1462).
That is the signature of a threshold defence -- collapse fires when hit hard, and the
attacker simply stays under it. Training only ever showed alphas in [0.2, 1.0] across broad
layer sets, i.e. large perturbations, so a threshold is exactly what it would have learned.

This sweeps alpha on the canonical rank-1 shape (shared direction, all layers) with
everything else fixed, so the only moving part is strength. alpha=1.0 must reproduce the
known rank-1 row (0.0000 harmful, GSM8K 0.0394) -- that is the built-in control.

    python experiments/version_a_alpha_sweep.py --checkpoint outputs/version_a_qwen_500.pt \
        --alpha 0.6 --out outputs/asweep_a060
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tamperforge import empirical_refusal_direction, load_model  # noqa: E402
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402
from train_tamper_resistant_v8 import _ablated_overrides  # noqa: E402
from v11_surgical_ablation import _load_trained  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--alpha", type=float, required=True,
                    help="ablation strength. 1.0 = full projection (the standard attack); "
                         ">1 over-projects past orthogonal, as Heretic does up to 1.5.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--direction-layer", type=int, default=20)
    ap.add_argument("--n-direction", type=int, default=256)
    ap.add_argument("--layers", default="all", help="'all' or 'lo-hi' band")
    ap.add_argument("--scope", choices=["all", "write_only", "read_only"], default="all",
                    help="which projections to ablate. write_only = (o_proj, down_proj), "
                         "which is exactly what Heretic touches and what training never "
                         "showed the model in isolation.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, tok, device = load_model(args.model_id, device=args.device)
    _load_trained(model, args.checkpoint)
    n_layers = len(model.model.layers)
    layers = (list(range(n_layers)) if args.layers == "all"
              else list(range(int(args.layers.split("-")[0]),
                              min(int(args.layers.split("-")[1]) + 1, n_layers))))

    harmful = load_advbench_prompts(None, n=args.n_direction, seed=42, source="walledai")
    benign = BENIGN_PROMPTS[: args.n_direction]
    with torch.no_grad():
        d = empirical_refusal_direction(model, tok, harmful, benign,
                                        args.direction_layer, device).to(device)
        # SCOPES["all"] read/write split, matching the canonical attack
        read_all = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "mlp.gate_proj", "mlp.up_proj")
        write_all = ("self_attn.o_proj", "mlp.down_proj")
        read_p = () if args.scope == "write_only" else read_all
        write_p = () if args.scope == "read_only" else write_all
        ov = _ablated_overrides(model, d, layers, read_p, write_p,
                                {li: args.alpha for li in layers})
        named = dict(model.named_parameters())
        for k, v in ov.items():
            named[k].copy_(v.to(named[k].dtype))

    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "alpha_sweep_meta.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "alpha": args.alpha, "scope": args.scope,
         "direction_layer": args.direction_layer, "n_layers_attacked": len(layers),
         "n_direction": args.n_direction}, indent=2))
    print(f"[asweep] alpha={args.alpha} dir_layer={args.direction_layer} "
          f"scope={args.scope} layers={len(layers)} -> {out}")


if __name__ == "__main__":
    main()
