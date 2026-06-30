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
from tamperforge.eval.judge import OpenRouterJudge
from tamperforge.eval.log import RunLogger, make_run_id


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
    harmful = 0
    refused = 0
    parse_fail = 0
    rating_counts: dict[str, int] = {}
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
            harmful += int(parsed.get("harmful_compliance") == 1)
            refused += int(parsed.get("refusal") == 1)
            parse_fail += int(parsed.get("harmful_compliance") is None or parsed.get("refusal") is None)
            rating = str(parsed.get("rating"))
            rating_counts[rating] = rating_counts.get(rating, 0) + 1
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

    n = len(judged)
    summary = {
        "run_id": run_id,
        "generations": str(args.generations),
        "judge_model": args.judge_model,
        "n": n,
        "judge_refusal_rate": refused / max(n, 1),
        "judge_asr": harmful / max(n, 1),
        "parse_failures": parse_fail,
        "rating_counts": rating_counts,
    }
    path = logger.summary(summary)
    print(json.dumps(summary, indent=2))
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
