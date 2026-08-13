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

from tamperforge import capture_residuals, empirical_refusal_direction, load_model  # noqa: E402
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402

READ_P = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
          "mlp.gate_proj", "mlp.up_proj")
WRITE_P = ("self_attn.o_proj", "mlp.down_proj")


def _load_trained(model, ckpt: str) -> None:
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    params = dict(model.named_parameters())
    n = 0
    for k, v in sd.items():
        if k in params and params[k].shape == v.shape:
            params[k].data.copy_(v.to(params[k].dtype))
            n += 1
    if n == 0:
        raise RuntimeError(
            f"checkpoint {ckpt} loaded zero compatible matrices into "
            f"{model.config._name_or_path}; check --model-id"
        )
    print(f"[surgical] loaded {n} trained matrices from {ckpt}")


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
def _refusal_subspace(model, tok, device, harmful, harmless, layer, rank) -> torch.Tensor:
    """Top-`rank` refusal directions by SVD of the harmful-minus-benign-mean matrix.

    The `rank_k_svd` attack from the zoo (docs/attack_zoo_v0.md #7), which exists to answer
    the "rank-1 is too weak an attack" objection (arXiv:2602.02132). Note this is a DIFFERENT
    estimator from `empirical_refusal_direction`'s difference-of-means: at rank 1 the two are
    close but not identical, so a k-sweep must use this for every k -- mixing mean-diff at
    k=1 with SVD at k>1 would confound the sweep with an estimator change.
    """
    Hh = capture_residuals(model, tok, harmful, [layer], device)[layer].float()
    Hb = capture_residuals(model, tok, harmless, [layer], device)[layer].float()
    D = Hh - Hb.mean(0, keepdim=True)
    _, s, Vh = torch.linalg.svd(D, full_matrices=False)
    frac = float(s[:rank].pow(2).sum() / s.pow(2).sum().clamp(min=1e-9))
    print(f"[surgical] refusal subspace rank {rank} captures {frac:.1%} of harmful-shift energy")
    return Vh[:rank].to(device)


@torch.no_grad()
def _attack_(model, R, layers) -> None:
    """Project the span of `R` (orthonormal rows, [k, d_model]) out of every attacked layer.

    Rank-1 reduces to the original outer-product form exactly:
    read  W - (W R^T) R  ==  W - outer(W @ d, d)
    write W - R^T (R W)  ==  W - outer(d, d @ W)
    """
    params = dict(model.named_parameters())
    RR = R.to(next(iter(params.values())).device)
    if RR.dim() == 1:
        RR = RR.unsqueeze(0)
    for li in layers:
        for name in READ_P:
            k = f"model.layers.{li}.{name}.weight"
            if k in params:
                W = params[k].float()
                params[k].copy_((W - (W @ RR.T) @ RR).to(params[k].dtype))
        for name in WRITE_P:
            k = f"model.layers.{li}.{name}.weight"
            if k in params:
                W = params[k].float()
                params[k].copy_((W - RR.T @ (RR @ W)).to(params[k].dtype))


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
                    help="dimension of the ablated refusal subspace. 1 = mean-diff direction "
                         "(the original behaviour); >1 = top-k SVD subspace (rank_k_svd).")
    ap.add_argument("--n-cap", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, tok, device = load_model(args.model_id, device=args.device)
    if args.checkpoint:
        _load_trained(model, args.checkpoint)
    layers = list(range(len(model.model.layers)))

    harmful = load_advbench_prompts(None, n=args.n_direction, seed=42, source="walledai")
    harmless = BENIGN_PROMPTS[: args.n_direction]
    # capture_residual returns CPU tensors, so d lands on CPU; the capability subspace
    # below is on-device. Move once here rather than at each use site.
    if args.refusal_rank == 1:
        R = empirical_refusal_direction(model, tok, harmful, harmless,
                                        args.direction_layer, device).to(device).unsqueeze(0)
    else:
        R = _refusal_subspace(model, tok, device, harmful, harmless,
                              args.direction_layer, args.refusal_rank)

    meta = {"model_id": args.model_id, "checkpoint": args.checkpoint,
            "direction_layer": args.direction_layer, "cap_rank": args.cap_rank,
            "refusal_rank": args.refusal_rank,
            "refusal_estimator": "mean_diff" if args.refusal_rank == 1 else "svd"}
    if args.cap_rank > 0:
        V = _cap_subspace(model, tok, device, _cap_prompts(args.n_cap),
                          args.direction_layer, args.cap_rank)
        R_cap = (R @ V.T) @ V                       # component of each basis vector in cap span
        # Energy fraction of the refusal subspace lying inside the capability span. At rank 1
        # this is exactly the old ||P_cap d|| / ||d||, so earlier numbers stay comparable.
        overlap = float(R_cap.norm() / R.norm().clamp(min=1e-9))
        R_s = R - R_cap
        if float(R_s.norm()) < 1e-4:
            raise SystemExit("refusal subspace lies (almost) entirely in the capability "
                             "span -- surgical ablation is undefined here, which would "
                             "itself be a strong entanglement result")
        # Re-orthonormalise: subtracting the capability component destroys orthonormality,
        # and _attack_'s projector is only idempotent for an orthonormal basis.
        # On CPU because MPS has no linalg_qr; the matrix is [d_model, k] so the transfer
        # is free next to the forward passes above.
        Q, _ = torch.linalg.qr(R_s.T.cpu())
        R_s = Q.T[: R.shape[0]].to(device)
        meta["cap_overlap_fraction"] = overlap
        if args.refusal_rank == 1:
            meta["cos_d_vs_dsurgical"] = float(
                (R[0] @ R_s[0]) / R[0].norm().clamp(min=1e-9))
            print(f"[surgical] {overlap:.1%} of d lies in the capability subspace; "
                  f"cos(d, d_surgical) = {meta['cos_d_vs_dsurgical']:.4f}")
        else:
            print(f"[surgical] {overlap:.1%} of the rank-{args.refusal_rank} refusal "
                  f"subspace energy lies in the capability subspace")
        R = R_s
    else:
        print(f"[surgical] cap-rank 0 -> plain rank-{args.refusal_rank} ablation (control arm)")

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
