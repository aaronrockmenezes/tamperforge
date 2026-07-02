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

from tamperforge import (empirical_refusal_direction, load_model,
                         orthonormalize_directions, svd_refusal_directions)
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts

READ = {"mlp": ("mlp.gate_proj", "mlp.up_proj"),
        "all": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "mlp.gate_proj", "mlp.up_proj")}
WRITE = {"mlp": ("mlp.down_proj",),
         "all": ("self_attn.o_proj", "mlp.down_proj")}


def _parse_layers(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    out: list[int] = []
    for c in spec.split(","):
        c = c.strip()
        if "-" in c:
            lo, hi = c.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif c:
            out.append(int(c))
    return list(dict.fromkeys(out))


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
def _attack(model, dirs, layers, scope) -> None:
    """Ablate a set of (orthonormal) directions from the scoped matrices."""
    dev = next(model.parameters()).device
    if dirs.dim() == 1:
        dirs = dirs.unsqueeze(0)
    dirs = [d.to(dev) for d in orthonormalize_directions(dirs)]
    for li in layers:
        layer = model.model.layers[li]
        for name in READ[scope]:
            mod = layer.get_submodule(name)
            W = mod.weight.data.float()
            for dd in dirs:
                W = W - torch.outer(W @ dd, dd)
            mod.weight.data = W.to(mod.weight.dtype)
        for name in WRITE[scope]:
            mod = layer.get_submodule(name)
            W = mod.weight.data.float()
            for dd in dirs:
                W = W - torch.outer(dd, dd @ W)
            mod.weight.data = W.to(mod.weight.dtype)
    print(f"[save] attacked: ablated {len(dirs)} dir(s) from {scope} of {len(layers)} layers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="P1b-A trained weights (.pt); omit for base")
    ap.add_argument("--out", required=True, help="output HF model dir")
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--attack", choices=["none", "mlp", "all"], default="none",
                    help="post-attack abliteration scope; 'none' saves the model as-is")
    ap.add_argument("--abliterate-layers", default="all",
                    help="all, or comma/range e.g. '13-25' (layer-subset attack, battery 1.2)")
    ap.add_argument("--direction-layer", type=int, default=13)
    ap.add_argument("--attack-estimator", choices=["diffmeans", "svd", "whitened_svd"],
                    default="diffmeans", help="refusal-direction estimator (battery 1.4)")
    ap.add_argument("--attack-rank", type=int, default=1,
                    help="number of directions to ablate (rank-k subspace, battery 1.3)")
    ap.add_argument("--direction-seed", type=int, default=42,
                    help="seed for the harmful/benign prompt sample used to estimate d")
    ap.add_argument("--n-direction", type=int, default=256)
    ap.add_argument("--per-layer", action="store_true",
                    help="ADAPTIVE attack: independent diffmeans refusal direction computed "
                         "at EACH layer and ablated from that layer (per-layer adaptive "
                         "abliteration; far stronger than one direction removed everywhere). "
                         "Ignores --attack-rank/estimator.")
    args = ap.parse_args()

    model, tok, device = load_model(args.model_id, args.device)
    if args.checkpoint:
        _load_trained(model, str(ROOT / args.checkpoint) if not Path(args.checkpoint).is_absolute()
                      else args.checkpoint)

    if args.attack != "none":
        layers = _parse_layers(args.abliterate_layers, len(model.model.layers))
        prompts = load_advbench_prompts(None, n=args.n_direction, seed=args.direction_seed,
                                        source="walledai")
        harmful = prompts[: args.n_direction]
        harmless = BENIGN_PROMPTS[: args.n_direction]
        with torch.no_grad():
            if args.per_layer:
                # ADAPTIVE: compute each layer's own refusal direction on the CLEAN weights
                # first (dict built before any ablation), then ablate each layer with its own.
                dir_by_layer = {li: empirical_refusal_direction(model, tok, harmful, harmless,
                                                                li, device) for li in layers}
                print(f"[save] per-layer adaptive attack: {len(layers)} independent directions")
                for li in layers:
                    _attack(model, dir_by_layer[li], [li], args.attack)
            elif args.attack_estimator == "diffmeans":
                dirs = empirical_refusal_direction(model, tok, harmful, harmless,
                                                   args.direction_layer, device)
                if args.attack_rank > 1:
                    print(f"[save] WARN: diffmeans is rank-1; --attack-rank {args.attack_rank} "
                          "ignored (use --attack-estimator svd for a rank-k subspace)")
                _attack(model, dirs, layers, args.attack)
            else:
                dirs = svd_refusal_directions(
                    model, tok, harmful, harmless, args.direction_layer, device,
                    k=args.attack_rank, whiten=(args.attack_estimator == "whitened_svd"))
                _attack(model, dirs, layers, args.attack)

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "p1b_meta.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "attack": args.attack,
        "attack_layers": args.abliterate_layers, "direction_layer": args.direction_layer,
        "attack_estimator": args.attack_estimator, "attack_rank": args.attack_rank,
        "direction_seed": args.direction_seed,
    }, indent=2))
    print(f"[save] wrote HF model dir -> {out}  (eval with p0_baseline_eval.py --backend vllm)")


if __name__ == "__main__":
    main()
