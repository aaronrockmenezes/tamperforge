#!/usr/bin/env python3
"""Judge completed Version G extended-suite artifacts while generation continues."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path("/workspace/tamperforge")
RESULTS = ROOT / "results"
SUITE = ROOT / "data/heldout_vg_20260804"
WORKERS = int(os.environ.get("VGHO_JUDGE_WORKERS", "96"))
YIELD_TO_MAIN = os.environ.get("VGHO_JUDGE_YIELD_TO_MAIN", "1") != "0"

MODEL_TAGS = [
    "qbase_clean", "qbase_rank1", "qg_clean", "qg_rank1", "qg_surg_k16",
    "qg_her_t73", "qg_benign_sft", "qg_benign_lora", "qg_harm_sft",
    "qg_harm_lora", "lbase_clean", "lbase_rank1", "lg_clean", "lg_rank1",
    "lg_surg_k16", "lg_her_t138",
]
ADAPT_TAGS = {"qg_benign_sft", "qg_benign_lora", "qg_harm_sft", "qg_harm_lora"}
HARM_BENCHES = ["harmbench", "jailbreakbench", "strongreject", "sorrybench", "beavertails"]


def row_count(path: Path) -> int:
    with path.open(errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def main_phase2_started() -> bool:
    logs = sorted((ROOT / "logs/eval").glob("version_g_extended_heldout_*.log"),
                  key=lambda path: path.stat().st_mtime)
    return bool(logs and "=== phase 2:" in logs[-1].read_text(errors="replace"))


def run_locked(rid: str, suffix: str, command: list[str]) -> bool:
    summary = RESULTS / f"{rid}_{suffix}" / "summary.json"
    if summary.exists():
        return False
    lock = RESULTS / f".{rid}_{suffix}.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        return False
    try:
        print(f"[async-judge] start {rid}_{suffix} workers={WORKERS}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        if not summary.exists():
            raise RuntimeError(f"missing summary after successful command: {summary}")
        print(f"[async-judge] complete {rid}_{suffix}", flush=True)
        return True
    finally:
        lock.rmdir()


def complete_generation(rid: str, bench: str, expected: dict[str, int]) -> Path | None:
    path = RESULTS / rid / "generations.jsonl"
    if not path.exists() or row_count(path) != expected[bench]:
        return None
    return path


def main() -> None:
    manifest = json.loads((SUITE / "manifest.json").read_text())
    expected = {name: row["frozen_n"] for name, row in manifest["benchmarks"].items()}
    judge_targets = [
        (tag, bench)
        for tag in MODEL_TAGS
        for bench in HARM_BENCHES + (["xstest_unsafe"] if tag in ADAPT_TAGS else [])
    ]
    overrefusal_targets = [
        (tag, bench) for tag in MODEL_TAGS if tag in ADAPT_TAGS
        for bench in ("xstest_safe", "orbench")
    ]

    while True:
        if YIELD_TO_MAIN and main_phase2_started():
            print("[async-judge] main launcher reached phase 2; yielding remaining work", flush=True)
            return

        progressed = False
        for tag, bench in judge_targets:
            rid = f"vgho_{tag}_{bench}"
            generations = complete_generation(rid, bench, expected)
            if generations is None:
                continue
            progressed |= run_locked(
                rid,
                "judged",
                [
                    "/venv/main/bin/python", "-u", "experiments/judge_generations.py",
                    "--generations", str(generations.relative_to(ROOT)),
                    "--run-id", f"{rid}_judged", "--num-workers", str(WORKERS),
                ],
            )
            if YIELD_TO_MAIN and main_phase2_started():
                print("[async-judge] main launcher reached phase 2; yielding remaining work", flush=True)
                return

        for tag, bench in overrefusal_targets:
            rid = f"vgho_{tag}_{bench}"
            generations = complete_generation(rid, bench, expected)
            if generations is None:
                continue
            progressed |= run_locked(
                rid,
                "orscore",
                [
                    "/venv/main/bin/python", "-u",
                    "scripts/external_benches/score_overrefusal.py",
                    "--generations", str(generations.relative_to(ROOT)),
                    "--run-id", rid, "--num-workers", str(WORKERS),
                ],
            )

        judged_done = sum(
            (RESULTS / f"vgho_{tag}_{bench}_judged" / "summary.json").exists()
            for tag, bench in judge_targets
        )
        or_done = sum(
            (RESULTS / f"vgho_{tag}_{bench}_orscore" / "summary.json").exists()
            for tag, bench in overrefusal_targets
        )
        print(
            f"[async-judge] checkpoint judged={judged_done}/{len(judge_targets)} "
            f"orscored={or_done}/{len(overrefusal_targets)}",
            flush=True,
        )
        if judged_done == len(judge_targets) and or_done == len(overrefusal_targets):
            print("[async-judge] all scoring complete", flush=True)
            return
        if not progressed:
            time.sleep(15)


if __name__ == "__main__":
    main()
