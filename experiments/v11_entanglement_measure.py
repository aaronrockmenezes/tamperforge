"""v11 gate 1: is safety actually ENTANGLED with capability, and by how much?

The whole TamperForge thesis is that removing the refusal direction should also remove
capability. The field has a name for the property being destroyed: **weight
disentanglement** [Ortiz-Jimenez et al., arXiv:2305.12827] -- distinct weight directions
governing non-overlapping function regions. TamperForge wants the opposite. Same quantity,
sign flipped.

Until now that has only ever been measured *indirectly*, by attacking a model and seeing
whether benchmarks fell. This measures it directly, on the weights, with no attack and no
generation. Three views:

  overlap   cos(d, u_i) between the empirical refusal direction and the top-k right
            singular directions of the capability-critical weight matrices. High = the
            refusal direction lives inside the subspace those matrices actually use, so
            projecting it out necessarily damages them.

  energy    Fraction of each matrix's spectral energy that the rank-1 abliteration edit
            removes: ||W - W_ablated||_F / ||W||_F. This is the amount of the layer's
            actual computation destroyed by the edit, independent of any benchmark.

  capgap    Teacher-forced CE gap on CAPABILITY data (chain-of-thought solutions) between
            clean and ablated weights. The direct weight-disentanglement error: does the
            refusal edit change behaviour in the capability region? This is the quantity
            gib_ce v2 would optimise, so measuring it first tells us the baseline and
            whether v7/v8 moved it at all.

Comparing base vs v7 vs v8 answers a question no previous run could: is the entanglement
real and increasing, or is the capability crater a side effect of generic weight damage?
If v8 shows higher overlap/capgap than base at the same energy, the mechanism is real and
this is a paper figure regardless of what v11 does next.

  python experiments/v11_entanglement_measure.py \
    --models base=Qwen/Qwen3-0.6B v8=outputs/qwen_v8_clean --direction-layer 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import (apply_chat_template_no_think, empirical_refusal_direction,  # noqa: E402
                         load_model)
from tamperforge.data import BENIGN_PROMPTS, load_advbench_prompts  # noqa: E402

# the matrices an all-scope abliteration actually edits
READ_P = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
          "mlp.gate_proj", "mlp.up_proj")
WRITE_P = ("self_attn.o_proj", "mlp.down_proj")


def _cap_texts(n: int) -> list[tuple[str, str]]:
    """(prompt, correct chain-of-thought answer) pairs -- the capability region.

    GSM8K solutions specifically: 'CE rose' on a correct multi-step derivation means the
    model can no longer produce that reasoning. That is a capability claim. 'CE rose' on
    wikitext prose is not, which is why the v5 prose-gap loss was uninformative.
    """
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    return [(r["question"], r["answer"]) for r in list(ds)[:n]]


@torch.no_grad()
def _cap_ce(model, tok, device, pairs) -> float:
    tot, n = 0.0, 0
    for q, a in pairs:
        prefix = apply_chat_template_no_think(tok, [{"role": "user", "content": q}],
                                              tokenize=False, add_generation_prompt=True)
        enc = tok(prefix + a, return_tensors="pt", truncation=True, max_length=512).to(device)
        plen = min(tok(prefix, return_tensors="pt")["input_ids"].shape[1],
                   enc["input_ids"].shape[1] - 1)
        labels = enc["input_ids"].clone()
        labels[:, :plen] = -100
        if (labels != -100).sum() == 0:
            continue
        tot += float(model(**enc, labels=labels).loss)
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def _ablate_(model, d, layers) -> None:
    """In-place rank-1 abliteration. Mutates the loaded model; reload to undo."""
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


@torch.no_grad()
def _overlap_and_energy(model, d, layers, topk: int) -> dict:
    """cos(d, top-k singular dirs) and the spectral energy the edit removes."""
    params = dict(model.named_parameters())
    dd = d.to(next(iter(params.values())).device).float()
    ov_read, ov_write, energy = [], [], []
    for li in layers:
        for name in READ_P + WRITE_P:
            k = f"model.layers.{li}.{name}.weight"
            if k not in params:
                continue
            W = params[k].float()
            is_read = name in READ_P
            # read matrices consume the residual on their INPUT side (columns), write
            # matrices emit into it on their OUTPUT side (rows) -- so the singular basis
            # that d must be compared against differs.
            M = W if is_read else W.T
            if M.shape[1] != dd.shape[0]:
                continue
            try:
                _, s, Vh = torch.linalg.svd(M, full_matrices=False)
            except Exception:  # noqa: BLE001
                continue
            k_ = min(topk, Vh.shape[0])
            # energy-weighted |cos| with the top-k right singular directions
            c = (Vh[:k_] @ dd).abs()
            w = s[:k_] / s[:k_].sum().clamp(min=1e-9)
            (ov_read if is_read else ov_write).append(float((c * w).sum()))
            abl = (W - torch.outer(W @ dd, dd)) if is_read else (W - torch.outer(dd, dd @ W))
            energy.append(float((W - abl).norm() / W.norm().clamp(min=1e-9)))
    m = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    return {"overlap_read": m(ov_read), "overlap_write": m(ov_write),
            "overlap_all": m(ov_read + ov_write), "edit_energy": m(energy)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, metavar="LABEL=PATH")
    ap.add_argument("--direction-layer", type=int, default=20)
    ap.add_argument("--n-direction", type=int, default=128)
    ap.add_argument("--n-cap", type=int, default=64, help="GSM8K pairs for the capgap")
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/v11_entanglement.json")
    args = ap.parse_args()

    harmful = load_advbench_prompts(None, n=args.n_direction, seed=42, source="walledai")
    harmless = BENIGN_PROMPTS[: args.n_direction]
    cap = _cap_texts(args.n_cap)
    print(f"[v11] {len(harmful)} harmful / {len(harmless)} harmless for d, "
          f"{len(cap)} GSM8K pairs for capgap")

    report = {}
    for spec in args.models:
        label, path = spec.split("=", 1)
        print(f"\n=== {label} ===")
        model, tok, device = load_model(path, device=args.device)
        layers = list(range(len(model.model.layers)))

        d = empirical_refusal_direction(model, tok, harmful, harmless,
                                        args.direction_layer, device)
        geo = _overlap_and_energy(model, d, layers, args.topk)
        ce_clean = _cap_ce(model, tok, device, cap)
        _ablate_(model, d, layers)
        ce_abl = _cap_ce(model, tok, device, cap)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gap = ce_abl - ce_clean
        # normalise the capability damage by how much of the weights the edit actually
        # removed -- otherwise a model that simply took a bigger hit looks 'more entangled'
        per_energy = gap / geo["edit_energy"] if geo["edit_energy"] else float("nan")
        report[label] = {**geo, "cap_ce_clean": ce_clean, "cap_ce_ablated": ce_abl,
                         "cap_gap": gap, "cap_gap_per_energy": per_energy}
        print(f"  overlap(d, top-{args.topk} sing dirs)  read {geo['overlap_read']:.4f} "
              f"write {geo['overlap_write']:.4f}  all {geo['overlap_all']:.4f}")
        print(f"  edit energy removed                  {geo['edit_energy']:.4f}")
        print(f"  capability CE   clean {ce_clean:.4f} -> ablated {ce_abl:.4f}"
              f"   GAP {gap:+.4f}")
        print(f"  cap_gap per unit edit energy         {per_energy:+.3f}   <-- THE NUMBER")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[v11] wrote {out}")
    print("READ: cap_gap_per_energy is the entanglement measure. Higher = removing the\n"
          "refusal direction costs more capability PER UNIT of weight actually edited.\n"
          "If v8 >> base, entanglement is real and trained-in, not generic weight damage.\n"
          "If v8 ~= base, the capability crater seen in evals is just collateral damage and\n"
          "the thesis has never actually been demonstrated at the weight level.")


if __name__ == "__main__":
    main()
