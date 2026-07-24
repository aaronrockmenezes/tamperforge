"""Option E gate: is compliance linearly readable from RESPONSE representations?

E (Circuit Breakers, arXiv:2406.04313) reroutes representations on harmful inputs away
from a compliance reference. Before wiring that as a loss, it needs the premise to hold:
a compliance direction must exist in representation space and separate coherent harm from
refusal.

Two deliberate differences from the option-F probe, which died reading checkpoint identity:

1. Representations are taken over **prompt + response**, not prompt alone. F showed
   compliance is not a prompt-conditioned property -- the model has not decided yet at the
   prompt's last token. CB operates on completions, so this matches it.
2. Everything is scored **within a single checkpoint**, on held-out prompts. A direction
   that merely identifies which model produced the text cannot score here.

Two questions:
  OWN      fit the direction on a model's own rows, evaluate on its held-out prompts.
           Is compliance linearly readable from response reps AT ALL? If no, E is dead.
  TRANSFER fit on model A, evaluate within model B. A loss needs a direction that is not
           specific to one checkpoint, so this is the number that matters for E as a loss.

  python experiments/gate_reroute.py --runs \
    heretic_mid=results/t0v2_heretic_kl0778_trial91 \
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

from tamperforge import apply_chat_template_no_think, capture_residuals, load_model  # noqa: E402
from tamperforge.eval.judge import usefulness_label  # noqa: E402

MODEL_OF = {
    "v8_clean": "outputs/qwen_v8_clean",
    "rank1": "outputs/qwen_v8_att_rank1",
    "heretic_weak": "outputs/attacked_snapshots/heretic_qwen_v8_kl0042_trial192",
    "heretic_mid": "outputs/attacked_snapshots/heretic_qwen_v8_kl0778_trial91",
    "heretic_strong": "outputs/attacked_snapshots/heretic_qwen_v8_kl2222_trial160",
}


def _auc(s: torch.Tensor, y: torch.Tensor) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r = torch.cat([pos, neg]).argsort().argsort().float()
    return ((r[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))).item()


def _rows(run: Path, tok) -> list[dict]:
    gens = {json.loads(x)["prompt"]: json.loads(x) for x in open(run / "generations.jsonl")}
    out = []
    for line in open(run.parent / (run.name + "_judged") / "judgments.jsonl"):
        r = json.loads(line)
        j = r.get("judge")
        if isinstance(j, str):
            j = ast.literal_eval(j)
        p = (j or {}).get("parsed") or {}
        if p.get("refusal") is None:
            continue
        lab = usefulness_label(p)
        if lab not in ("harmful_actionable", "refused"):
            continue  # compliance direction is harm-vs-refusal; gibberish is a third thing
        g = gens.get(r["prompt"])
        if not g or not (g.get("response") or "").strip():
            continue
        prefix = apply_chat_template_no_think(tok, [{"role": "user", "content": r["prompt"]}],
                                              tokenize=False, add_generation_prompt=True)
        out.append({"prompt": r["prompt"], "text": prefix + g["response"],
                    "y": int(lab == "harmful_actionable")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, metavar="LABEL=RESULTS_DIR")
    ap.add_argument("--holdout", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/gate_reroute.json")
    args = ap.parse_args()

    data = {}
    for spec in args.runs:
        label, rdir = spec.split("=", 1)
        model, tok, device = load_model(MODEL_OF[label], device=args.device)
        rows = _rows(ROOT / rdir, tok)
        n_pos = sum(r["y"] for r in rows)
        print(f"[E] {label}: {len(rows)} rows ({n_pos} harmful / {len(rows)-n_pos} refused)")
        layers = list(range(len(model.model.layers)))
        with torch.no_grad():
            res = capture_residuals(model, tok, [r["text"] for r in rows], layers, device,
                                    use_chat_template=False)
        data[label] = {"res": res, "y": torch.tensor([r["y"] for r in rows]).float(),
                       "prompts": [r["prompt"] for r in rows]}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {}
    labels = list(data)
    for li in sorted(next(iter(data.values()))["res"]):
        line = {}
        dirs = {}
        for lb in labels:
            X, y = data[lb]["res"][li], data[lb]["y"]
            uniq = sorted(set(data[lb]["prompts"]))
            perm = torch.randperm(len(uniq), generator=torch.Generator().manual_seed(args.seed))
            test_p = {uniq[i] for i in perm[: int(len(uniq) * args.holdout)].tolist()}
            te = torch.tensor([p in test_p for p in data[lb]["prompts"]])
            tr = ~te
            if y[tr].sum() < 5 or (1 - y[tr]).sum() < 5:
                continue
            # compliance direction = mean(harmful) - mean(refused), unit-norm
            d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
            d = d / d.norm().clamp(min=1e-8)
            dirs[lb] = d
            line[f"own:{lb}"] = _auc(X[te] @ d, y[te])
        for src in dirs:
            for dst in labels:
                if src == dst or dst not in data:
                    continue
                X, y = data[dst]["res"][li], data[dst]["y"]
                line[f"xfer:{src}->{dst}"] = _auc(X @ dirs[src], y)
        report[li] = line
        own = [v for k, v in line.items() if k.startswith("own:") and v == v]
        xfr = [v for k, v in line.items() if k.startswith("xfer:") and v == v]
        print(f"  L{li:<3d} OWN {'/'.join(f'{v:.3f}' for v in own) or '-':22s} "
              f"TRANSFER {'/'.join(f'{v:.3f}' for v in xfr) or '-'}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({str(k): v for k, v in report.items()}, indent=2))
    print(f"\n[E] wrote {out}")
    print("GATE: OWN >> 0.5 means compliance IS linearly readable from response reps (the\n"
          "premise E needs). TRANSFER is the one that matters for a loss -- a direction that\n"
          "only works on the checkpoint it was fitted on is the option-F failure again.")


if __name__ == "__main__":
    main()
