#!/usr/bin/env python
"""Replay a logged Heretic trial as an AttackSpec and export the attacked model.

THIS IS THE CONTROL GATE FOR version_C. Before any training run, replaying t99 against
version_B s500 must reproduce its known result (0.3212 harmful / GSM8K 0.3700, per
results/version_b_final_2026_08_01/summary.json). If it does not, the tent implementation
is wrong and every downstream number is worthless -- four separate harness bugs this
session produced plausible-looking wrong numbers, and controls are what caught all of them.

    python experiments/version_c_replay.py --checkpoint outputs/version_b_qwen_500.pt \
        --trial t99 --out outputs/vcreplay_t99
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

from tamperforge import empirical_refusal_directions, load_model  # noqa: E402
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402
from train_tamper_resistant_v8 import _ablated_overrides  # noqa: E402
from v11_surgical_ablation import _load_trained  # noqa: E402
from version_a_attack import HERETIC_PROJ, heretic_spec, tent_weight  # noqa: E402
from version_c_loop import (HERETIC_BAD, HERETIC_GOOD, HERETIC_SYSTEM,  # noqa: E402
                            heretic_directions)

SUMMARY = ROOT / "results/version_b_final_2026_08_01/summary.json"

@torch.no_grad()
def heretic_delta(W: torch.Tensor, v: torch.Tensor, weight: float,
                  r: int = 3, seed: int = 0) -> torch.Tensor:
    """Heretic's actual weight delta, `row_normalization = FULL`.

    Transcribed from heretic/model.py:551-597. The ablation is computed against
    ROW-NORMALIZED weights, the result is renormalized and rescaled to the original row
    magnitudes, and the resulting delta is then approximated by a rank-`r` randomized SVD
    (it has to fit in a LoRA adapter). None of that is the plain
    `W - a * outer(v, v @ W)` we train against.

    Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
    """
    W_org = W
    rn = W.norm(dim=1, keepdim=True).clamp(min=1e-12)
    Wn = W / rn
    lora_A = (v @ Wn).view(1, -1)
    lora_B = (-weight * v).view(-1, 1)
    Wa = Wn + lora_B @ lora_A
    Wa = Wa / Wa.norm(dim=1, keepdim=True).clamp(min=1e-12)
    Wa = Wa * rn
    D = Wa - W_org
    # svd_lowrank is randomized; heretic reseeds immediately before the call. q and niter
    # are heretic's (2r+4, 6), which is accurate enough that the seed barely matters here.
    torch.manual_seed(seed)
    U, S, V = torch.svd_lowrank(D, q=2 * r + 4, niter=6)
    return (U[:, :r] * S[:r]) @ V[:, :r].T


@torch.no_grad()
def svd_directions(model, attacked_dir: str, layers, device) -> dict[int, torch.Tensor]:
    """Recover heretic's ACTUAL per-layer directions from its saved weights.

    For a write-projection ablation dW = a * outer(d, d^T W), which is rank-1, so d is the
    left singular vector of dW. Sign is irrelevant: outer(d, d^T W) is invariant under
    d -> -d. Measured spectral mass of the top component is 0.99+, so the recovery is clean.

    This gives cos(d_recovered, d_heretic) = 1 by construction, which is the point: it
    separates "tent/application math wrong" from "direction estimate off". If the replay
    still fails to reproduce t99 with these, the fault is in how we APPLY the attack, not
    in how we estimate the direction.
    """
    from safetensors.torch import load_file

    att = load_file(str(Path(attacked_dir) / "model.safetensors"))
    clean = {k: v.detach().float().cpu() for k, v in model.named_parameters()}
    out: dict[int, torch.Tensor] = {}
    agree = []
    for li in layers:
        cand = {}
        for nm in HERETIC_PROJ:
            k = f"model.layers.{li}.{nm}.weight"
            if k not in att:
                continue
            dw = clean[k] - att[k].float()
            if dw.norm() < 1e-6:
                continue
            U, S, _ = torch.linalg.svd(dw.double(), full_matrices=False)
            cand[nm] = (U[:, 0], (S[0] ** 2 / (S ** 2).sum()).item())
        if not cand:
            continue
        # both projections at a layer share one direction in heretic; if they disagree the
        # rank-1 recovery is unsound and everything after this is meaningless.
        if len(cand) == 2:
            a, b = (v[0] for v in cand.values())
            agree.append(abs(torch.dot(a, b).item()))
        best = max(cand.values(), key=lambda v: v[1])
        out[li] = best[0].float().to(device)
    if agree:
        print(f"[svd] o_proj-vs-down_proj direction agreement: "
              f"min {min(agree):.4f} median {sorted(agree)[len(agree) // 2]:.4f}")
    return out


@torch.no_grad()
def _heretic_full_overrides(model, d: dict, spec) -> dict:
    """Apply heretic's FULL row-normalized delta per (layer, projection)."""
    params = dict(model.named_parameters())
    ov = {}
    for li in spec.layers:
        for nm in spec.write_proj + spec.read_proj:
            a = spec.alphas.get(nm, {}).get(li, 0.0) if isinstance(
                next(iter(spec.alphas.values())), dict) else spec.alphas[li]
            if a <= 0.0:
                continue
            key = f"model.layers.{li}.{nm}.weight"
            W = params[key].float()
            ov[key] = (W + heretic_delta(W, d[li].to(W.device), a)).to(params[key].dtype)
    return ov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--checkpoint", default=None,
                    help="trained .pt; omit to replay the attack against the BASE model "
                         "(needed for the heretic-vs-base ceiling control)")
    ap.add_argument("--trial", required=True, help="t17 | t99 | t65")
    ap.add_argument("--params-json", default=str(SUMMARY))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-direction", type=int, default=256)
    ap.add_argument("--application", choices=["plain", "heretic_full"], default="plain",
                    help="'plain' = W - a*outer(d, d@W), what we train against. "
                         "'heretic_full' = heretic's row_normalization=FULL delta.")
    ap.add_argument("--svd-from", default=None,
                    help="attacked model dir to recover heretic's true directions from "
                         "(requires --direction-recipe svd)")
    ap.add_argument("--direction-recipe", choices=["ours", "heretic", "svd"], default="heretic",
                    help="'heretic' reproduces its own direction pipeline (own datasets, "
                         "system prompt, projected abliteration). 'ours' uses walledai + "
                         "BENIGN_PROMPTS, which does NOT reproduce heretic's attack.")
    ap.add_argument("--override-direction-index", default=None,
                    help="replace the trial's direction_index (float, or 'per layer'). "
                         "Sweeping THIS while holding the tents/scope/application fixed "
                         "isolates the layer axis at heretic's real operating point -- the "
                         "old dl_sweep varied it under a different attack shape entirely "
                         "(flat alpha, all layers, plain application).")
    ap.add_argument("--dir-no-thinking", action="store_true",
                    help="force thinking off when reading directions; heretic leaves the "
                         "Qwen3 default (ON), so this should normally stay unset")
    ap.add_argument("--no-orthogonalize", action="store_true",
                    help="disable projected abliteration (heretic defaults it ON)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    trials = json.loads(Path(args.params_json).read_text())["heretic_trials"]
    params = trials[args.trial]

    model, tok, device = load_model(args.model_id, device=args.device)
    if args.checkpoint:
        _load_trained(model, args.checkpoint)
    else:
        print("[replay] no checkpoint -- attacking the BASE model")
    n_layers = len(model.model.layers)

    if args.override_direction_index is not None:
        params = dict(params)
        params["direction_index"] = args.override_direction_index
    spec = heretic_spec(params, n_layers, tag=f"heretic:replay:{args.trial}")
    print(f"[replay] {args.trial} per_layer={spec.per_layer} read_layer={spec.read_layer} "
          f"layers={spec.layers[0]}-{spec.layers[-1]} n={len(spec.layers)}")
    for proj, prof in spec.alphas.items():
        if prof:
            ls = sorted(prof)
            print(f"   {proj:<18} layers {ls[0]}-{ls[-1]} ({len(ls)}) "
                  f"peak {max(prof.values()):.3f} floor {min(prof.values()):.3f}")

    if args.direction_recipe == "svd":
        if not args.svd_from:
            ap.error("--direction-recipe svd requires --svd-from")
        d = svd_directions(model, args.svd_from, spec.layers, device)
        missing = [L for L in spec.layers if L not in d]
        assert not missing, f"no direction recovered for layers {missing}"
        with torch.no_grad():
            if args.application == "heretic_full":
                ov = _heretic_full_overrides(model, d, spec)
            else:
                ov = _ablated_overrides(model, d, spec.layers, spec.read_proj,
                                        spec.write_proj, spec.alphas)
            named = dict(model.named_parameters())
            for k, v in ov.items():
                named[k].copy_(v.to(named[k].dtype))
        print(f"[replay] overrode {len(ov)} parameter tensors (SVD-recovered directions)")
        _save(model, tok, args, params, spec, ov)
        return

    if args.direction_recipe == "heretic":
        need = spec.layers if spec.per_layer else sorted(
            {int(spec.read_layer), min(int(spec.read_layer) + 1, n_layers - 1)})
        hd = heretic_directions(model, tok, need, device,
                                orthogonalize=not args.no_orthogonalize,
                                thinking=not args.dir_no_thinking)
        if spec.per_layer:
            d = hd
        else:
            import math
            frac, lo = math.modf(float(spec.read_layer))
            lo = int(lo)
            a, b = hd[lo], hd[min(lo + 1, n_layers - 1)]
            d = torch.lerp(a, b, frac)
            d = d / d.norm().clamp(min=1e-9)
        with torch.no_grad():
            if args.application == "heretic_full":
                dd = d if isinstance(d, dict) else {L: d for L in spec.layers}
                ov = _heretic_full_overrides(model, dd, spec)
            else:
                ov = _ablated_overrides(model, d, spec.layers, spec.read_proj,
                                        spec.write_proj, spec.alphas)
            named = dict(model.named_parameters())
            for k, v in ov.items():
                named[k].copy_(v.to(named[k].dtype))
        print(f"[replay] overrode {len(ov)} tensors "
              f"(heretic direction recipe, application={args.application})")
        _save(model, tok, args, params, spec, ov)
        return

    harmful = load_advbench_prompts(None, n=args.n_direction, seed=42, source="walledai")
    benign = BENIGN_PROMPTS[: args.n_direction]
    with torch.no_grad():
        if spec.per_layer:
            src = empirical_refusal_directions(model, tok, harmful, benign,
                                               spec.layers, device)
            d = {li: v.to(device) for li, v in src.items()}
        else:
            # fractional shared direction: lerp neighbours, then renormalise (heretic's math)
            import math
            frac, lo = math.modf(float(spec.read_layer))
            lo = int(lo)
            need = sorted({lo, min(lo + 1, n_layers - 1)})
            got = empirical_refusal_directions(model, tok, harmful, benign, need, device)
            a = got[lo].to(device)
            b = got[min(lo + 1, n_layers - 1)].to(device)
            d = torch.lerp(a, b, frac)
            d = d / d.norm().clamp(min=1e-9)

        ov = _ablated_overrides(model, d, spec.layers, spec.read_proj,
                                spec.write_proj, spec.alphas)
        named = dict(model.named_parameters())
        for k, v in ov.items():
            named[k].copy_(v.to(named[k].dtype))
    print(f"[replay] overrode {len(ov)} parameter tensors")
    _save(model, tok, args, params, spec, ov)


def _save(model, tok, args, params, spec, ov) -> None:
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "replay_meta.json").write_text(json.dumps(
        {"checkpoint": args.checkpoint, "trial": args.trial, "heretic_params": params,
         "direction_recipe": args.direction_recipe,
         "application": args.application,
         "svd_from": args.svd_from,
         "orthogonalize": not args.no_orthogonalize,
         "direction_thinking": not args.dir_no_thinking,
         "override_direction_index": args.override_direction_index,
         "per_layer": spec.per_layer, "read_layer": spec.read_layer,
         "n_layers_touched": len(spec.layers), "n_overrides": len(ov)}, indent=2))
    print(f"[replay] -> {out}")


def _selfcheck() -> None:
    from version_a_attack import Tent
    t = Tent(max_weight=1.11, max_pos=26.33, min_weight=0.01, min_dist=5.39)
    assert abs(tent_weight(t, 26.33) - 1.11) < 1e-9, "peak wrong"
    assert tent_weight(t, 26.33 + 5.39 * 1.001) == 0.0, "no hard cutoff"
    # just inside the edge -> min_weight. (Exactly max_pos+min_dist is unreachable in
    # float; heretic cuts on `distance > min_dist`, matched in tent_weight.)
    assert abs(tent_weight(t, 26.33 + 5.39 * 0.999) - 0.01) < 2e-3, "edge != min_weight"
    mid = tent_weight(t, 26.33 + 2.695)
    assert abs(mid - (1.11 + 0.01) / 2) < 1e-6, f"not linear: {mid}"
    assert tent_weight(t, 0) == 0.0, "far layer must be untouched"
    # t99's down_proj must touch only the top of the stack, o_proj must be broad
    p = json.loads(SUMMARY.read_text())["heretic_trials"]["t99"]
    spec = heretic_spec(p, 28)
    dn = sorted(spec.alphas["mlp.down_proj"])
    op = sorted(spec.alphas["self_attn.o_proj"])
    assert dn[0] >= 20, f"down_proj should be top-only, got {dn[0]}-{dn[-1]}"
    assert len(op) > len(dn), "o_proj should be broader than down_proj"
    assert spec.per_layer and spec.read_proj == (), "t99 is per-layer, write-only"
    print(f"selfcheck OK: t99 down_proj {dn[0]}-{dn[-1]} ({len(dn)}), "
          f"o_proj {op[0]}-{op[-1]} ({len(op)})")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
