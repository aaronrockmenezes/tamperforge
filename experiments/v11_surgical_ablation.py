"""v11 gate 2: surgical refusal ablation -- the attack built to defeat entanglement.

Surgical Refusal Ablation [arXiv:2601.08489] observes that the raw refusal direction has
non-trivial cosine similarity with capability directions ("Logic", "Coding"), and removes
those components BEFORE ablating. It does not delete d; it deletes

    d_surgical = normalise( d - P_cap d )

where P_cap projects onto a capability subspace. So the attacker strips exactly the part of
the refusal direction that TamperForge relies on to carry the damage.

This is the direct counter to the entire TamperForge thesis, it is listed as an arXiv
priority in MEMORY.md's attack zoo, and it has never been run. If it uncensors v8 without
the capability crater, the defence has a bigger problem than any loss term -- and that has
to be known BEFORE more training, not after.

The capability subspace is estimated here from activations on capability data (GSM8K
chain-of-thought, code, instruction following), by SVD of the benign-centred activation
matrix. That is the cheap attacker-side estimate; the alignment-tax literature
(arXiv:2602.07892 orthogonal gradient projection, arXiv:2512.11391 null-space PO) computes
essentially the same subspace in order to AVOID it, so a real attacker has better estimators
available off the shelf. Treat this as a lower bound on the attack.

Writes attacked HF model dirs; evaluate them with the normal p0_baseline_eval + judge path.

  python experiments/v11_surgical_ablation.py \
    --model-id Qwen/Qwen3-0.6B --checkpoint outputs/adapters/..._v8.pt \
    --direction-layer 20 --cap-rank 16 --out outputs/v11_v8_surgical_k16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import (RANK_ESTIMATORS, abliterate_model_inplace, capture_residuals,
                         decoder_layers, load_model, load_partial_checkpoint,
                         refusal_subspaces_from_activations)  # noqa: E402
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402


def _load_trained(model, ckpt: str) -> None:
    load_partial_checkpoint(model, ckpt)


def _cap_prompts(n: int) -> list[str]:
    """Capability-region inputs: math reasoning + instruction following."""
    out: list[str] = []
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="train")
        out += [r["question"] for r in list(ds)[: n // 2]]
    except Exception as e:  # noqa: BLE001
        print(f"[surgical] gsm8k unavailable ({type(e).__name__}), using benign only")
    out += BENIGN_PROMPTS[: n - len(out)]
    return out[:n]


@torch.no_grad()
def _cap_subspace(model, tok, device, prompts, layer, rank) -> torch.Tensor:
    """Top-`rank` right singular directions of centred capability activations."""
    H = capture_residuals(model, tok, prompts, [layer], device)[layer].float()
    H = H - H.mean(0, keepdim=True)
    _, s, Vh = torch.linalg.svd(H, full_matrices=False)
    keep = Vh[:rank]
    frac = float(s[:rank].pow(2).sum() / s.pow(2).sum().clamp(min=1e-9))
    print(f"[surgical] capability subspace rank {rank} captures {frac:.1%} of variance")
    return keep.to(device)


@torch.no_grad()
def _refusal_subspace(model, tok, device, harmful, harmless, layer, rank,
                      estimator="arditi_residual") -> torch.Tensor:
    """Estimate one rank-k refusal basis with the explicitly selected estimator."""
    Hh = capture_residuals(model, tok, harmful, [layer], device)[layer].float()
    Hb = capture_residuals(model, tok, harmless, [layer], device)[layer].float()
    R = refusal_subspaces_from_activations(Hh, Hb, (rank,), estimator)[rank]
    print(f"[surgical] refusal rank {rank} estimator={estimator}")
    return R.to(device)


@torch.no_grad()
def _attack_(model, R, layers) -> None:
    """Project the span of `R` (orthonormal rows, [k, d_model]) out of every attacked layer.

    Rank-1 reduces to the original outer-product form exactly:
    read  W - (W R^T) R  ==  W - outer(W @ d, d)
    write W - R^T (R W)  ==  W - outer(d, d @ W)
    """
    abliterate_model_inplace(model, R, layers)


def _load_selected_basis(path: str, key: str, args, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload.get("_meta", {})
    try:
        entry = payload["bases"][key]
    except KeyError as exc:
        raise SystemExit(f"basis key {key!r} missing from {path}") from exc
    expected = {
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
    }
    for field, value in expected.items():
        if meta.get(field) != value:
            raise SystemExit(f"basis {field} mismatch: {meta.get(field)!r} != {value!r}")
    entry_expected = {
        "layer": args.direction_layer,
        "attack_rank": args.refusal_rank,
        "capability_rank": args.cap_rank,
    }
    for field, value in entry_expected.items():
        if entry.get(field) != value:
            raise SystemExit(f"basis {field} mismatch: {entry.get(field)!r} != {value!r}")
    R = entry["basis"].float()
    if R.ndim != 2 or R.shape[0] != args.refusal_rank:
        raise SystemExit(f"invalid basis shape {tuple(R.shape)} for rank {args.refusal_rank}")
    gram = R @ R.T
    if not torch.allclose(gram, torch.eye(R.shape[0]), atol=2e-4, rtol=2e-4):
        raise SystemExit("selected attack basis is not orthonormal")
    return R.to(device), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--checkpoint", default=None, help="trained .pt; omit for base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--direction-layer", type=int, default=20)
    ap.add_argument("--n-direction", type=int, default=256)
    ap.add_argument("--cap-rank", type=int, default=16, help="0 = plain ablation control")
    ap.add_argument("--refusal-rank", type=int, default=1,
                    help="dimension of the ablated refusal subspace; estimator controls how "
                         "all basis vectors are constructed")
    ap.add_argument("--rank-estimator", choices=RANK_ESTIMATORS, default="arditi_residual",
                    help="arditi_residual keeps rank-1 exactly equal to mean-difference and "
                         "adds orthogonal residual SVD directions for higher ranks")
    ap.add_argument("--basis-file", default=None,
                    help="selected_bases.pt from adaptive_attack_sweep; skips recomputation")
    ap.add_argument("--basis-key", default=None,
                    help="variant key inside --basis-file")
    ap.add_argument("--n-cap", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if bool(args.basis_file) != bool(args.basis_key):
        ap.error("--basis-file and --basis-key must be supplied together")

    model, tok, device = load_model(args.model_id, device=args.device)
    if args.checkpoint:
        _load_trained(model, args.checkpoint)
    layers = list(range(len(decoder_layers(model))))

    meta = {"model_id": args.model_id, "checkpoint": args.checkpoint,
            "direction_layer": args.direction_layer, "cap_rank": args.cap_rank,
            "refusal_rank": args.refusal_rank,
            "refusal_estimator": args.rank_estimator}
    if args.basis_file:
        R, basis_meta = _load_selected_basis(args.basis_file, args.basis_key, args, device)
        meta.update({"basis_file": args.basis_file, "basis_key": args.basis_key,
                     "basis_checkpoint_sha256": basis_meta.get("checkpoint_sha256")})
        print(f"[attack] exact selected basis {args.basis_key} from {args.basis_file}")
    else:
        harmful = load_advbench_prompts(None, n=args.n_direction, seed=42, source="walledai")
        harmless = BENIGN_PROMPTS[: args.n_direction]
        R = _refusal_subspace(model, tok, device, harmful, harmless,
                              args.direction_layer, args.refusal_rank, args.rank_estimator)
        if args.cap_rank > 0:
            V = _cap_subspace(model, tok, device, _cap_prompts(args.n_cap),
                              args.direction_layer, args.cap_rank)
            R_cap = (R @ V.T) @ V
            overlap = float(R_cap.norm() / R.norm().clamp(min=1e-9))
            R_s = R - R_cap
            if float(R_s.norm()) < 1e-4:
                raise SystemExit("refusal subspace lies almost entirely in capability span")
            Q, _ = torch.linalg.qr(R_s.T.cpu())
            R = Q.T[: R.shape[0]].to(device)
            meta["cap_overlap_fraction"] = overlap
            print(f"[surgical] removed capability overlap={overlap:.1%}")
        else:
            print(f"[surgical] cap-rank 0 -> plain rank-{args.refusal_rank} ablation")

    _attack_(model, R, layers)
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "v11_surgical_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[surgical] wrote {out}")
    print("NEXT: p0_baseline_eval (--qwen-thinking default --max-new-tokens 512) + judge.\n"
          "The comparison that matters is surgical vs the cap-rank 0 control on the SAME\n"
          "checkpoint: if surgical gets coherent harm where rank-1 got gibberish, the\n"
          "entanglement defence is bypassable by a one-line projection.")


if __name__ == "__main__":
    main()
