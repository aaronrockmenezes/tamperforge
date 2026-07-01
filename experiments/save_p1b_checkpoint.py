#!/usr/bin/env python
"""Save a P1b-A model (optionally post-attack) as a full HF dir for vLLM.

The P1b-A model is pure weights (no hook), so once saved with save_pretrained it
runs through vLLM like any Gemma checkpoint — batched, fast. Use this to
materialize the eval conditions as real model dirs, then eval with
`p0_baseline_eval.py --backend vllm --model-id <dir>` and judge.

Examples:
  # trained model, no attack (the product)
  python experiments/save_p1b_checkpoint.py \
    --checkpoint outputs/tamper_resistant_p1b_v6.pt --attack none \
    --out outputs/p1b_v6_clean

  # scope-MISMATCH attack: we trained scope=all, attacker ablates MLP only
  python experiments/save_p1b_checkpoint.py \
    --checkpoint outputs/tamper_resistant_p1b_v6.pt --attack mlp \
    --out outputs/p1b_v6_att_mlp

  # attacker uses a different direction layer
  python experiments/save_p1b_checkpoint.py \
    --checkpoint outputs/tamper_resistant_p1b_v6.pt --attack all --direction-layer 9 \
    --out outputs/p1b_v6_att_L9

  # base control, attacked (no --checkpoint)
  python experiments/save_p1b_checkpoint.py --attack all --out outputs/p1b_v6_base_att
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

READ = {"mlp": ("mlp.gate_proj", "mlp.up_proj"),
        "all": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "mlp.gate_proj", "mlp.up_proj")}
WRITE = {"mlp": ("mlp.down_proj",),
         "all": ("self_attn.o_proj", "mlp.down_proj")}


def _load_trained(model, ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.pop("_meta", {})
    named = dict(model.named_parameters())
    n = 0
    for name, tensor in ckpt.items():
        if name in named:
            named[name].data.copy_(tensor.to(named[name].dtype).to(named[name].device))
            n += 1
        else:
            raise RuntimeError(f"checkpoint tensor not in model: {name}")
    print(f"[save] loaded {n} trained matrices")
    return meta


@torch.no_grad()
def _attack(model, d, layers, scope) -> None:
    dev = next(model.parameters()).device
    dd = (d / d.norm().clamp(min=1e-8)).to(dev)
    for li in layers:
        layer = model.model.layers[li]
        for name in READ[scope]:
            mod = layer.get_submodule(name)
            W = mod.weight.data.float()
            mod.weight.data = (W - torch.outer(W @ dd, dd)).to(mod.weight.dtype)
        for name in WRITE[scope]:
            mod = layer.get_submodule(name)
            W = mod.weight.data.float()
            mod.weight.data = (W - torch.outer(dd, dd @ W)).to(mod.weight.dtype)
    print(f"[save] attacked: ablated emp. refusal dir from {scope} of {len(layers)} layers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="P1b-A trained weights (.pt); omit for base")
    ap.add_argument("--out", required=True, help="output HF model dir")
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--attack", choices=["none", "mlp", "all"], default="none",
                    help="post-attack abliteration scope; 'none' saves the model as-is")
    ap.add_argument("--abliterate-layers", default="all")
    ap.add_argument("--direction-layer", type=int, default=13)
    ap.add_argument("--n-direction", type=int, default=256)
    args = ap.parse_args()

    model, tok, device = load_model(args.model_id, args.device)
    if args.checkpoint:
        _load_trained(model, str(ROOT / args.checkpoint) if not Path(args.checkpoint).is_absolute()
                      else args.checkpoint)

    if args.attack != "none":
        n_layers = len(model.model.layers)
        layers = list(range(n_layers)) if args.abliterate_layers == "all" \
            else [int(x) for x in args.abliterate_layers.split(",")]
        prompts = load_advbench_prompts(None, n=args.n_direction, seed=42, source="walledai")
        with torch.no_grad():
            d = empirical_refusal_direction(model, tok, prompts[: args.n_direction],
                                            BENIGN_PROMPTS[: args.n_direction],
                                            args.direction_layer, device)
        _attack(model, d, layers, args.attack)

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "p1b_meta.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "attack": args.attack,
        "attack_layers": args.abliterate_layers, "direction_layer": args.direction_layer,
    }, indent=2))
    print(f"[save] wrote HF model dir -> {out}  (eval with p0_baseline_eval.py --backend vllm)")


if __name__ == "__main__":
    main()
