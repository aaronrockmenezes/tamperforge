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
def _attack_(model, d, layers) -> None:
    params = dict(model.named_parameters())
    dd = d.to(next(iter(params.values())).device)
    for li in layers:
        for name in READ_P:
            k = f"model.layers.{li}.{name}.weight"
            if k in params:
                W = params[k].float()
                params[k].copy_((W - torch.outer(W @ dd, dd)).to(params[k].dtype))
        for name in WRITE_P:
            k = f"model.layers.{li}.{name}.weight"
            if k in params:
                W = params[k].float()
                params[k].copy_((W - torch.outer(dd, dd @ W)).to(params[k].dtype))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--checkpoint", default=None, help="trained .pt; omit for base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--direction-layer", type=int, default=20)
    ap.add_argument("--n-direction", type=int, default=256)
    ap.add_argument("--cap-rank", type=int, default=16, help="0 = plain rank-1 control")
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
    d = empirical_refusal_direction(model, tok, harmful, harmless,
                                    args.direction_layer, device).to(device)

    meta = {"model_id": args.model_id, "checkpoint": args.checkpoint,
            "direction_layer": args.direction_layer, "cap_rank": args.cap_rank}
    if args.cap_rank > 0:
        V = _cap_subspace(model, tok, device, _cap_prompts(args.n_cap),
                          args.direction_layer, args.cap_rank)
        d_cap = V.T @ (V @ d)                       # component of d inside capability span
        overlap = float(d_cap.norm() / d.norm().clamp(min=1e-9))
        d_s = d - d_cap
        nrm = d_s.norm()
        if float(nrm) < 1e-4:
            raise SystemExit("refusal direction lies (almost) entirely in the capability "
                             "span -- surgical ablation is undefined here, which would "
                             "itself be a strong entanglement result")
        d_s = d_s / nrm
        meta["cap_overlap_fraction"] = overlap
        meta["cos_d_vs_dsurgical"] = float((d @ d_s) / d.norm().clamp(min=1e-9))
        print(f"[surgical] {overlap:.1%} of d lies in the capability subspace; "
              f"cos(d, d_surgical) = {meta['cos_d_vs_dsurgical']:.4f}")
        d = d_s
    else:
        print("[surgical] cap-rank 0 -> plain rank-1 ablation (control arm)")

    _attack_(model, d, layers)
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
