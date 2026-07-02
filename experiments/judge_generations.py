#!/usr/bin/env python
"""Judge an existing generations.jsonl without rerunning local model eval."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge.eval.env import load_dotenv
from tamperforge.eval.judge import OpenRouterJudge, usefulness_label
from tamperforge.eval.log import RunLogger, make_run_id


def _empty_counts() -> dict:
    return {"n": 0, "harmful": 0, "refused": 0, "parse_failures": 0, "rating_counts": {},
            "usefulness_counts": {}}


def _add_count(counts: dict, parsed: dict) -> None:
    counts["n"] += 1
    counts["harmful"] += int(parsed.get("harmful_compliance") == 1)
    counts["refused"] += int(parsed.get("refusal") == 1)
    counts["parse_failures"] += int(
        parsed.get("harmful_compliance") is None or parsed.get("refusal") is None
    )
    rating = str(parsed.get("rating"))
    counts["rating_counts"][rating] = counts["rating_counts"].get(rating, 0) + 1
    label = usefulness_label(parsed)
    counts["usefulness_counts"][label] = counts["usefulness_counts"].get(label, 0) + 1


def _finish_counts(counts: dict) -> dict:
    n = counts["n"]
    return {
        "n": n,
        "judge_refusal_rate": counts["refused"] / max(n, 1),
        "judge_asr": counts["harmful"] / max(n, 1),
        "parse_failures": counts["parse_failures"],
        "rating_counts": counts["rating_counts"],
        # distinguishes "actually harmless" (refused/benign) from "harmless because
        # broken" (gibberish) — the ASR-alone failure mode from the tamper-resistance
        # runs. harmful_actionable_rate is the STRICTER metric for the paper.
        "usefulness_counts": counts["usefulness_counts"],
        "harmful_actionable_rate": counts["usefulness_counts"].get("harmful_actionable", 0) / max(n, 1),
        "gibberish_rate": counts["usefulness_counts"].get("gibberish", 0) / max(n, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--judge-model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--judge-json-mode", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--judge-max-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--row-ids", default=None, help="Comma-separated source row i values to judge")
    ap.add_argument("--condition", default=None,
                    help="Only judge rows with this condition value.")
    ap.add_argument("--group-by", default="condition",
                    help="Also emit grouped summaries for this row field; use '' to disable.")
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    run_id = args.run_id or make_run_id("judge_generations")
    logger = RunLogger(ROOT / args.out_dir, run_id, repo_root=ROOT)
    logger.write_manifest({"script": "judge_generations.py", "args": vars(args)})
    judge = OpenRouterJudge(
        args.judge_model,
        json_mode=args.judge_json_mode,
        max_tokens=args.judge_max_tokens,
    )

    rows = []
    with Path(args.generations).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit is not None:
        rows = rows[: args.limit]
    if args.row_ids:
        keep = {int(x.strip()) for x in args.row_ids.split(",") if x.strip()}
        rows = [row for row in rows if int(row["i"]) in keep]
    if args.condition:
        rows = [row for row in rows if row.get("condition") == args.condition]

    def score_one(item):
        idx, row = item
        try:
            result = judge.score(row["prompt"], row["response"])
        except Exception as exc:  # keep batch alive; count as parse failure
            result = {
                "judge_model": args.judge_model,
                "raw": {"error": repr(exc)},
                "parsed": {
                    "refusal": None,
                    "harmful_compliance": None,
                    "rating": None,
                    "reason": repr(exc),
                },
            }
        return idx, row, result

    judged = []
    overall = _empty_counts()
    grouped: dict[str, dict] = {}
    done = 0
    work = list(enumerate(rows, start=1))
    with ThreadPoolExecutor(max_workers=max(args.num_workers, 1)) as ex:
        futures = [ex.submit(score_one, item) for item in work]
        pbar = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="judge",
            dynamic_ncols=True,
        )
        for fut in pbar:
            i, row, result = fut.result()
            done += 1
            parsed = result["parsed"]
            _add_count(overall, parsed)
            if args.group_by:
                group = str(row.get(args.group_by, "__missing__"))
                grouped.setdefault(group, _empty_counts())
                _add_count(grouped[group], parsed)
            out = {**row, "judge": result}
            judged.append(out)
            logger.judgment(out)
            logger.event("judge_progress", {"done": done, "total": len(rows), "source_index": i})
            pbar.set_postfix(
                src=i,
                refusal=parsed.get("refusal"),
                harmful=parsed.get("harmful_compliance"),
                rating=parsed.get("rating"),
            )

    summary = {
        "run_id": run_id,
        "generations": str(args.generations),
        "judge_model": args.judge_model,
        **_finish_counts(overall),
    }
    if args.group_by:
        summary[f"by_{args.group_by}"] = {
            group: _finish_counts(counts) for group, counts in sorted(grouped.items())
        }
    path = logger.summary(summary)
    print(json.dumps(summary, indent=2))
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
