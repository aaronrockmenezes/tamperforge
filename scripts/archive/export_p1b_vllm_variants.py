#!/usr/bin/env python
"""Export P1b checkpoints into HF model dirs for vLLM/lm-eval.

The P1b trainer saves only changed matrices. vLLM needs a full HF model dir.
This script loads Gemma, overlays the P1b checkpoint, optionally applies the
real empirical-refusal abliteration attack, then save_pretrained().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import empirical_refusal_direction, load_model
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts

READ = {
    "mlp": ("mlp.gate_proj", "mlp.up_proj"),
    "all": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "mlp.gate_proj", "mlp.up_proj"),
}
WRITE = {
    "mlp": ("mlp.down_proj",),
    "all": ("self_attn.o_proj", "mlp.down_proj"),
}


def _parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return list(dict.fromkeys(out))


def _load_trained(model, checkpoint: Path) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu")
    meta = ckpt.pop("_meta", {})
    named = dict(model.named_parameters())
    expected = set(meta.get("trainable", ckpt.keys()))
    missing = []
    loaded = 0
    for name, tensor in ckpt.items():
        if name not in named:
            missing.append(name)
            continue
        named[name].data.copy_(tensor.to(named[name].dtype).to(named[name].device))
        loaded += 1
    if missing:
        raise RuntimeError(f"checkpoint tensors not in model: {missing[:5]}")
    if set(ckpt) != expected:
        raise RuntimeError("checkpoint tensors do not match _meta.trainable")
    if loaded != len(expected):
        raise RuntimeError(f"loaded {loaded}, expected {len(expected)}")
    print(f"[load] {loaded} trained matrices from {checkpoint}")
    return meta


@torch.no_grad()
def _attack(model, d: torch.Tensor, layers: list[int], scope: str) -> None:
    dev = next(model.parameters()).device
    dd = (d / d.norm().clamp(min=1e-8)).to(dev)
    for li in layers:
        layer = model.model.layers[li]
        for name in READ[scope]:
            mod = layer.get_submodule(name)
            w = mod.weight.data.float()
            mod.weight.data = (w - torch.outer(w @ dd, dd)).to(mod.weight.dtype)
        for name in WRITE[scope]:
            mod = layer.get_submodule(name)
            w = mod.weight.data.float()
            mod.weight.data = (w - torch.outer(dd, dd @ w)).to(mod.weight.dtype)
    print(f"[attack] ablated empirical refusal dir from {scope} across {len(layers)} layers")


def _save(model, tok, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "tamperforge_export_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[saved] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-root", default="outputs/p1b_v6_vllm")
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--variants", default="trained_clean,trained_attacked,base_attacked",
                    help="Comma list: trained_clean,trained_attacked,base_attacked")
    ap.add_argument("--attack-scope", choices=["mlp", "all"], default="all")
    ap.add_argument("--abliterate-layers", default="all")
    ap.add_argument("--direction-layer", type=int, default=13)
    ap.add_argument("--n-direction", type=int, default=256)
    ap.add_argument("--advbench-source", choices=["walledai", "local"], default="walledai")
    ap.add_argument("--advbench-split", default="train")
    ap.add_argument("--advbench-csv", default=str(ROOT / "data" / "advbench_harmful_behaviors.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    valid = {"trained_clean", "trained_attacked", "base_attacked"}
    bad = sorted(set(variants) - valid)
    if bad:
        raise SystemExit(f"unknown variants: {bad}")

    out_root = ROOT / args.out_root
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    prompts = load_advbench_prompts(
        args.advbench_csv,
        n=args.n_direction,
        seed=args.seed,
        source=args.advbench_source,
        split=args.advbench_split,
    )
    harmless = BENIGN_PROMPTS[: min(args.n_direction, len(BENIGN_PROMPTS))]

    if "trained_clean" in variants or "trained_attacked" in variants:
        model, tok, device = load_model(args.model_id, args.device)
        ckpt_meta = _load_trained(model, ckpt_path)
        layers = _parse_layers(args.abliterate_layers, len(model.model.layers))
        base_meta = {
            "model_id": args.model_id,
            "checkpoint": str(ckpt_path),
            "checkpoint_meta": ckpt_meta,
            "attack_scope": args.attack_scope,
            "layers": layers,
            "direction_layer": args.direction_layer,
            "n_direction": args.n_direction,
        }
        if "trained_clean" in variants:
            _save(model, tok, out_root / "trained_clean", {**base_meta, "variant": "trained_clean"})
        if "trained_attacked" in variants:
            d = empirical_refusal_direction(
                model, tok, prompts, harmless, args.direction_layer, device
            )
            _attack(model, d, layers, args.attack_scope)
            _save(model, tok, out_root / "trained_attacked",
                  {**base_meta, "variant": "trained_attacked"})
        del model

    if "base_attacked" in variants:
        model, tok, device = load_model(args.model_id, args.device)
        layers = _parse_layers(args.abliterate_layers, len(model.model.layers))
        d = empirical_refusal_direction(model, tok, prompts, harmless, args.direction_layer, device)
        _attack(model, d, layers, args.attack_scope)
        _save(model, tok, out_root / "base_attacked", {
            "model_id": args.model_id,
            "variant": "base_attacked",
            "attack_scope": args.attack_scope,
            "layers": layers,
            "direction_layer": args.direction_layer,
            "n_direction": args.n_direction,
        })


if __name__ == "__main__":
    main()
