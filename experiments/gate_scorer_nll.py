"""Option D gate: does a frozen scorer LM tell gibberish from coherent text?

D scores the ATTACKED model's own generation under a frozen reference LM. Gibberish
should be implausible (high NLL); fluent text, harmful or not, should be plausible
(low NLL). Unlike A (fixed reference text) and F (prompt-only probe), this reads the
actual output, so representational drift cannot fake it.

This gate needs no generation: every checkpoint's outputs are already on disk. It
scores existing generations.jsonl rows under the frozen base and asks the only
question that matters, entirely WITHIN one checkpoint so model identity is not a
confound:

    within a single model, does scorer-NLL separate its gibberish rows from its
    coherent rows?

That is the property D claims. If AUC ~= 0.5 here, D is dead too and the frozen
scorer is measuring nothing a loss could use.

Also reports coherent-harmful vs coherent-refusal. D is NOT expected to separate
those -- it is a coherence signal, not a harm signal -- and it must not, or it would
just be smuggling in the safety-tuned scorer's own opinion. Reported so the claim
stays honest either way.

  python experiments/gate_scorer_nll.py --scorer Qwen/Qwen3-0.6B \
    --runs v8_clean=results/t0v2_qwen_v8_clean \
           rank1=results/t0v2_qwen_v8_att_rank1 \
           heretic_strong=results/t0v2_heretic_kl2222_trial160
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import apply_chat_template_no_think, load_model  # noqa: E402
from tamperforge.eval.judge import usefulness_label  # noqa: E402


def _auc(scores: list[float], y: list[int]) -> float:
    s = torch.tensor(scores)
    t = torch.tensor(y)
    pos, neg = s[t == 1], s[t == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = torch.cat([pos, neg]).argsort().argsort().float()
    return ((r[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))).item()


def _rows(run: Path) -> list[dict]:
    """Generations joined to judge labels by row index."""
    gens = [json.loads(x) for x in open(run / "generations.jsonl")]
    jpath = run.parent / (run.name + "_judged") / "judgments.jsonl"
    labels: dict[str, str] = {}
    if jpath.exists():
        for line in open(jpath):
            r = json.loads(line)
            j = r.get("judge")
            if isinstance(j, str):
                j = ast.literal_eval(j)
            p = (j or {}).get("parsed") or {}
            if p.get("refusal") is not None:
                labels[r["prompt"]] = usefulness_label(p)
    for g in gens:
        g["label"] = labels.get(g.get("prompt"), None)
    return [g for g in gens if g.get("label")]


@torch.no_grad()
def _nll(model, tok, device, prompt: str, response: str) -> float:
    """Mean per-token NLL of *response* given *prompt*, under the frozen scorer."""
    prefix = apply_chat_template_no_think(tok, [{"role": "user", "content": prompt}],
                                          tokenize=False, add_generation_prompt=True)
    full = tok(prefix + response, return_tensors="pt", truncation=True, max_length=1024).to(device)
    plen = tok(prefix, return_tensors="pt")["input_ids"].shape[1]
    labels = full["input_ids"].clone()
    labels[:, :plen] = -100
    if (labels != -100).sum() == 0:
        return float("nan")
    return float(model(**full, labels=labels).loss)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, metavar="LABEL=RESULTS_DIR")
    ap.add_argument("--scorer", default="Qwen/Qwen3-0.6B",
                    help="frozen reference LM. Use the BASE pretrain, not a safety-tuned "
                         "model, so harmful-but-fluent still reads as fluent.")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/gate_scorer_nll.json")
    args = ap.parse_args()

    model, tok, device = load_model(args.scorer, device=args.device)
    model.eval()
    report = {}
    for spec in args.runs:
        label, rdir = spec.split("=", 1)
        rows = _rows(ROOT / rdir if not Path(rdir).is_absolute() else Path(rdir))
        if args.limit > 0:
            rows = rows[: args.limit]
        for r in rows:
            r["nll"] = _nll(model, tok, device, r["prompt"], r.get("response") or "")
        rows = [r for r in rows if r["nll"] == r["nll"]]
        by = {}
        for r in rows:
            by.setdefault(r["label"], []).append(r["nll"])
        means = {k: sum(v) / len(v) for k, v in sorted(by.items())}

        gib = [r for r in rows if r["label"] == "gibberish"]
        coh = [r for r in rows if r["label"] in ("harmful_actionable", "refused", "benign")]
        auc_gib = _auc([r["nll"] for r in gib + coh], [1] * len(gib) + [0] * len(coh))

        harm = [r for r in rows if r["label"] == "harmful_actionable"]
        ref = [r for r in rows if r["label"] == "refused"]
        auc_harm = _auc([r["nll"] for r in harm + ref], [1] * len(harm) + [0] * len(ref))

        report[label] = {"n": len(rows), "mean_nll_by_label": means,
                         "auc_gibberish_vs_coherent": auc_gib,
                         "auc_harmful_vs_refused": auc_harm,
                         "counts": {k: len(v) for k, v in by.items()}}
        print(f"\n=== {label} (n={len(rows)}) ===")
        for k, v in means.items():
            print(f"  {k:20s} n={len(by[k]):4d}  mean NLL {v:6.3f}")
        print(f"  AUC gibberish-vs-coherent  {auc_gib:.3f}   <-- THE GATE (want >> 0.5)")
        print(f"  AUC harmful-vs-refused     {auc_harm:.3f}   (expected ~0.5; D is a "
              f"coherence signal, not a harm signal)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[gate] wrote {out}")


if __name__ == "__main__":
    main()
