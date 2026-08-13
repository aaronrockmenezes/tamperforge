#!/usr/bin/env python3
"""Replace one failed judgment with a successful single-row retry and recompute summary."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiments.judge_generations import _add_count, _empty_counts, _finish_counts


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-dir", type=Path, required=True)
    ap.add_argument("--repair-dir", type=Path, required=True)
    args = ap.parse_args()

    main_path = args.main_dir / "judgments.jsonl"
    repair_path = args.repair_dir / "judgments.jsonl"
    rows = read_jsonl(main_path)
    repairs = read_jsonl(repair_path)
    if len(repairs) != 1:
        raise SystemExit(f"expected exactly one repair row, got {len(repairs)}")
    repair = repairs[0]
    repair_i = repair.get("i")
    matches = [idx for idx, row in enumerate(rows) if row.get("i") == repair_i]
    if len(matches) != 1:
        raise SystemExit(f"expected one main row with i={repair_i}, got {len(matches)}")
    parsed = repair.get("judge", {}).get("parsed", {})
    if parsed.get("harmful_compliance") is None or parsed.get("refusal") is None:
        raise SystemExit("repair judgment is still unparsed")

    shutil.copy2(main_path, args.main_dir / f"judgments.pre_repair_i{repair_i}.jsonl")
    summary_path = args.main_dir / "summary.json"
    shutil.copy2(summary_path, args.main_dir / f"summary.pre_repair_i{repair_i}.json")
    rows[matches[0]] = repair
    main_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    overall = _empty_counts()
    grouped: dict[str, dict] = {}
    for row in rows:
        parsed = row["judge"]["parsed"]
        _add_count(overall, parsed)
        group = str(row.get("condition", "__missing__"))
        grouped.setdefault(group, _empty_counts())
        _add_count(grouped[group], parsed)

    summary = json.loads(summary_path.read_text())
    summary.update(_finish_counts(overall))
    summary["by_condition"] = {
        group: _finish_counts(counts) for group, counts in sorted(grouped.items())
    }
    summary["repair"] = {
        "source": str(args.repair_dir),
        "row_i": repair_i,
        "backup_judgments": f"judgments.pre_repair_i{repair_i}.jsonl",
        "backup_summary": f"summary.pre_repair_i{repair_i}.json",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n": summary["n"], "parse_failures": summary["parse_failures"],
                      "row_i": repair_i}, indent=2))


if __name__ == "__main__":
    main()
