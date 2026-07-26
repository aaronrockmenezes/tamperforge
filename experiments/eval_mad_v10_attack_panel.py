#!/usr/bin/env python
"""Evaluate MAD-v10 checkpoints against a fixed attack panel.

This is teacher-forced capability eval only. It never prints prompts or targets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tamperforge import empirical_refusal_directions, load_model  # noqa: E402
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402
from tamperforge.eval.log import make_run_id  # noqa: E402
from train_mad_v10 import (  # noqa: E402
    _attack_panel,
    _parse_layers,
    _target_ce,
    load_capability_pairs,
)
from train_tamper_resistant_v8 import _ablated_overrides  # noqa: E402


def _apply_checkpoint(model, path: Path) -> list[str]:
    ckpt = torch.load(path, map_location="cpu")
    named = dict(model.named_parameters())
    loaded = []
    with torch.no_grad():
        for name, value in ckpt.items():
            if name.startswith("_"):
                continue
            if name not in named:
                continue
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
            loaded.append(name)
    if not loaded:
        raise ValueError(f"no model weights found in checkpoint {path}")
    return loaded


def _safe_ckpt_name(path: Path) -> str:
    return path.name.replace(".pt", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="outputs/hf_qwen/Qwen3-0.6B")
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--qwen-thinking", choices=["off", "on", "default"],
                    default=os.environ.get("TF_QWEN_THINKING", "off"))
    ap.add_argument("--cap-datasets", default="gsm8k,arc,tiny_if")
    ap.add_argument("--n-cap-eval", type=int, default=64)
    ap.add_argument("--max-target-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attack-layers", default="10-27")
    ap.add_argument("--panel", choices=["heretic", "full"], default="heretic")
    ap.add_argument("--n-direction", type=int, default=256)
    ap.add_argument("--n-harmful-direction", type=int, default=520)
    ap.add_argument("--n-benign-direction", type=int, default=1000)
    args = ap.parse_args()

    os.environ["TF_QWEN_THINKING"] = args.qwen_thinking
    run_id = make_run_id("mad_v10_panel")
    out = Path(args.out) if args.out else ROOT / "results" / run_id / "attack_panel.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[panel] loading eval rows: {args.cap_datasets}", flush=True)
    cap_eval = load_capability_pairs(args.cap_datasets, args.n_cap_eval, args.seed + 1, "test", False)
    harmful_for_direction = load_advbench_prompts(
        ROOT / "data" / "advbench_harmful_behaviors.csv",
        n=args.n_harmful_direction,
        seed=args.seed,
        source="local",
    )
    from tamperforge.data_p1b import load_benign_instructions

    benign_for_direction = load_benign_instructions(args.n_benign_direction, seed=args.seed)

    model, tok, device = load_model(args.model_id, args.device)
    model.eval()
    n_layers = len(model.model.layers)
    attack_layers = _parse_layers(args.attack_layers, n_layers)
    specs = _attack_panel(n_layers, attack_layers, args.panel)

    with out.open("w") as f:
        for ckpt_str in args.checkpoints:
            ckpt_path = ROOT / ckpt_str
            if not ckpt_path.exists():
                print(f"[panel] missing {ckpt_path}", flush=True)
                continue
            loaded = _apply_checkpoint(model, ckpt_path)
            print(f"[panel] checkpoint={ckpt_path.name} loaded_matrices={len(loaded)}", flush=True)
            hs = harmful_for_direction[: min(args.n_direction, len(harmful_for_direction))]
            bs = benign_for_direction[: min(args.n_direction, len(benign_for_direction))]
            with torch.no_grad():
                direction_layers = sorted(set(attack_layers + [n_layers // 2]))
                d_by_layer = empirical_refusal_directions(model, tok, hs, bs, direction_layers, device)
                d = d_by_layer[n_layers // 2]
                clean_eval = float(_target_ce(model, tok, cap_eval, device, max_len=args.max_target_len))
                gaps = []
                for tag, rp, wp, layers, alphas, per_layer in specs:
                    overrides = _ablated_overrides(model, d_by_layer if per_layer else d, layers, rp, wp, alphas)
                    attacked_eval = float(
                        _target_ce(model, tok, cap_eval, device, overrides=overrides, max_len=args.max_target_len)
                    )
                    row = {
                        "checkpoint": _safe_ckpt_name(ckpt_path),
                        "attack_panel_tag": tag,
                        "clean_cap_eval_ce": clean_eval,
                        "attacked_cap_eval_ce": attacked_eval,
                        "cap_eval_gap": attacked_eval - clean_eval,
                    }
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    gaps.append((tag, row["cap_eval_gap"]))
            best = max(gaps, key=lambda x: x[1])
            avg = sum(g for _, g in gaps) / max(len(gaps), 1)
            print(f"[panel] {ckpt_path.name} clean={clean_eval:.3f} avg_gap={avg:.3f} best={best[0]}:{best[1]:.3f}", flush=True)

    print(f"[panel] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
