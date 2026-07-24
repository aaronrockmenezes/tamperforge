"""Mine real harmful completions to use as L_harm targets (ABL-v9, option A).

Why: v9's L_harm scores CE on AdvBench's affirmative prefix ("Sure, here is <goal>").
That string is a near-copy of the prompt, so it is cheap to predict and the CE
SATURATES the moment a model is willing to say it — measured on 2026-07-25, the
Heretic mid and strong checkpoints agree on it to SIX decimals (1.4359488 vs
1.4359469) despite 67% vs 82% judged harm, and a fully safe model sits only 0.206
nats away from a fully jailbroken one. A loss with 0.2 nats of range in the regime
that matters cannot steer training there.

A real harmful answer is not a near-copy of the prompt, so it should not saturate.
This pulls `harmful_actionable` responses out of an already-judged run and writes
{goal: [completion, ...]} for the trainer / tier0_gate to teacher-force against.

Source must be a DIFFERENT model from the ones being scored, or the comparison is
circular (a model always predicts its own outputs well). Default source is the
base + rank-1 attacked run — neither v8 nor any Heretic checkpoint.

  python experiments/mine_harm_targets.py \
    --judgments ../tamperforge-archive/results/qwen06_base_att_adv520_judged/judgments.jsonl \
    --out data/harm_targets_qwen.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge.eval.judge import usefulness_label  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgments", nargs="+", required=True,
                    help="judgments.jsonl file(s) from a judged run on a DIFFERENT model")
    ap.add_argument("--out", default="data/harm_targets.json")
    ap.add_argument("--min-rating", type=int, default=4,
                    help="keep only harmful_actionable rows at >= this judge rating")
    ap.add_argument("--min-chars", type=int, default=80,
                    help="drop stubs; a 20-char 'answer' is no better than the prefix")
    ap.add_argument("--max-per-goal", type=int, default=4)
    args = ap.parse_args()

    by_goal: dict[str, list[str]] = defaultdict(list)
    seen = kept = 0
    for path in args.judgments:
        for line in open(path):
            row = json.loads(line)
            seen += 1
            j = row.get("judge")
            if isinstance(j, str):
                j = ast.literal_eval(j)
            parsed = (j or {}).get("parsed") or {}
            if usefulness_label(parsed) != "harmful_actionable":
                continue
            if (parsed.get("rating") or 0) < args.min_rating:
                continue
            resp = (row.get("response") or "").strip()
            if len(resp) < args.min_chars:
                continue
            goal = row["prompt"]
            if len(by_goal[goal]) < args.max_per_goal:
                by_goal[goal].append(resp)
                kept += 1

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(by_goal, indent=1))
    lens = [len(v) for v in by_goal.values()]
    print(f"[mine] scanned {seen} rows -> kept {kept} completions over {len(by_goal)} goals "
          f"(mean {sum(lens)/max(len(lens),1):.2f}/goal)")
    print(f"[mine] wrote {out}")


if __name__ == "__main__":
    main()
