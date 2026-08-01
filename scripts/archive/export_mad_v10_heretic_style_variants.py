#!/usr/bin/env python
"""Export matched MAD-v10 Heretic-style attacked models for vLLM.

The attack matches the S175 training distribution:
  - all write projections (attention o_proj and MLP down_proj)
  - layers 10-27
  - one empirical refusal direction per layer
  - one held-out partial strength per layer, sampled uniformly from 0.2-0.6

The strength schedule and direction-prompt sample are identical for the trained
and base variants. Directions are recomputed on each model, as an adaptive
attacker would do.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import empirical_refusal_directions, load_model  # noqa: E402
from tamperforge.data import load_advbench_prompts  # noqa: E402
from tamperforge.data_p1b import load_benign_instructions  # noqa: E402

WRITE_PROJECTIONS = ("self_attn.o_proj", "mlp.down_proj")


def _parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    layers: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            layers.extend(range(int(lo), int(hi) + 1))
        else:
            layers.append(int(chunk))
    layers = list(dict.fromkeys(layers))
    invalid = [li for li in layers if li < 0 or li >= n_layers]
    if invalid:
        raise ValueError(f"invalid layers for {n_layers}-layer model: {invalid}")
    return layers


def _load_checkpoint(model, checkpoint: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu")
    meta = state.get("_meta", {})
    named = dict(model.named_parameters())
    loaded = 0
    with torch.no_grad():
        for name, value in state.items():
            if name.startswith("_"):
                continue
            if name not in named:
                raise KeyError(f"checkpoint tensor not found in model: {name}")
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
            loaded += 1
    if loaded == 0:
        raise RuntimeError(f"no model tensors loaded from {checkpoint}")
    print(f"[load] checkpoint={checkpoint.name} matrices={loaded}", flush=True)
    return meta


@torch.no_grad()
def _apply_attack(
    model,
    directions: dict[int, torch.Tensor],
    layers: list[int],
    alphas: dict[int, float],
) -> None:
    device = next(model.parameters()).device
    for layer_index in layers:
        direction = directions[layer_index].to(device)
        direction = direction / direction.norm().clamp(min=1e-8)
        alpha = alphas[layer_index]
        layer = model.model.layers[layer_index]
        for module_name in WRITE_PROJECTIONS:
            module = layer.get_submodule(module_name)
            weight = module.weight.data.float()
            module.weight.data = (
                weight - alpha * torch.outer(direction, direction @ weight)
            ).to(module.weight.dtype)


def _save(model, tokenizer, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    (out / "tamperforge_export_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True)
    )
    print(f"[saved] {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", default="outputs/hf_qwen/Qwen3-0.6B")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attack-layers", default="10-27")
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=0.6)
    parser.add_argument("--attack-seed", type=int, default=20260726)
    parser.add_argument("--direction-seed", type=int, default=20260928)
    parser.add_argument("--n-direction", type=int, default=256)
    parser.add_argument("--n-harmful-direction", type=int, default=520)
    parser.add_argument("--n-benign-direction", type=int, default=1000)
    parser.add_argument(
        "--variants",
        default="trained_heretic_attacked,base_heretic_attacked",
    )
    args = parser.parse_args()

    variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    valid = {"trained_heretic_attacked", "base_heretic_attacked"}
    if not variants or variants - valid:
        raise SystemExit(f"variants must be a subset of {sorted(valid)}")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    harmful = load_advbench_prompts(
        ROOT / "data" / "advbench_harmful_behaviors.csv",
        n=args.n_harmful_direction,
        seed=args.direction_seed,
        source="local",
    )
    benign = load_benign_instructions(args.n_benign_direction, seed=args.direction_seed)
    direction_rng = random.Random(args.direction_seed)
    harmful_sample = direction_rng.sample(harmful, min(args.n_direction, len(harmful)))
    benign_sample = direction_rng.sample(benign, min(args.n_direction, len(benign)))

    attack_rng = random.Random(args.attack_seed)
    alpha_schedule: dict[int, float] | None = None

    for variant in ("trained_heretic_attacked", "base_heretic_attacked"):
        if variant not in variants:
            continue
        model, tokenizer, device = load_model(args.model_id, args.device)
        checkpoint_meta = None
        if variant == "trained_heretic_attacked":
            checkpoint_meta = _load_checkpoint(model, checkpoint)

        layers = _parse_layers(args.attack_layers, len(model.model.layers))
        if alpha_schedule is None:
            alpha_schedule = {
                layer_index: attack_rng.uniform(args.alpha_min, args.alpha_max)
                for layer_index in layers
            }
        directions = empirical_refusal_directions(
            model,
            tokenizer,
            harmful_sample,
            benign_sample,
            layers,
            device,
        )
        _apply_attack(model, directions, layers, alpha_schedule)
        _save(
            model,
            tokenizer,
            out_root / variant,
            {
                "variant": variant,
                "model_id": args.model_id,
                "checkpoint": str(checkpoint) if checkpoint_meta is not None else None,
                "checkpoint_step": (
                    checkpoint_meta.get("step") if checkpoint_meta is not None else None
                ),
                "attack": {
                    "profile": "partial_perlayer",
                    "write_scope": "all_write",
                    "write_projections": list(WRITE_PROJECTIONS),
                    "layers": layers,
                    "alpha_min": args.alpha_min,
                    "alpha_max": args.alpha_max,
                    "alphas": {str(key): value for key, value in alpha_schedule.items()},
                    "attack_seed": args.attack_seed,
                    "direction_seed": args.direction_seed,
                    "n_direction": len(harmful_sample),
                    "direction_method": "empirical_refusal_directions",
                },
            },
        )
        del directions, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
