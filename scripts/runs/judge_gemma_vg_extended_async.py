#!/usr/bin/env python3
"""Score complete Gemma held-out artifacts while generation continues."""
from __future__ import annotations
import json, subprocess, time
from pathlib import Path

ROOT = Path("/workspace/tamperforge")
RESULTS = ROOT / "results"
SUITE = ROOT / "data/heldout_vg_20260804"
TAGS = ["gbase_clean", "gg_clean", "gg_her"]
HARM = ["harmbench", "jailbreakbench", "strongreject", "sorrybench", "beavertails", "xstest_unsafe"]
OVER = ["xstest_safe", "orbench"]
WORKERS = 96

def rows(path: Path) -> int:
    with path.open(errors="replace") as h:
        return sum(bool(x.strip()) for x in h)

def complete(rid: str, bench: str, expected: dict[str, int]) -> Path | None:
    p = RESULTS / rid / "generations.jsonl"
    return p if p.exists() and rows(p) == expected[bench] else None

def run(rid: str, suffix: str, cmd: list[str]) -> bool:
    summary = RESULTS / f"{rid}_{suffix}" / "summary.json"
    if summary.exists(): return False
    lock = RESULTS / f".{rid}_{suffix}.lock"
    try: lock.mkdir()
    except FileExistsError: return False
    try:
        print(f"[gemma-judge] start {rid}_{suffix} workers={WORKERS}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        if not summary.exists(): raise RuntimeError(f"missing {summary}")
        return True
    finally:
        lock.rmdir()

def main() -> None:
    if (RESULTS / ".defer_api_scoring").exists():
        print("[gemma-judge] deferred: OpenRouter quota marker present", flush=True)
        return
    manifest = json.loads((SUITE / "manifest.json").read_text())
    expected = {k: v["frozen_n"] for k, v in manifest["benchmarks"].items()}
    jt = [(t,b) for t in TAGS for b in HARM]
    ot = [(t,b) for t in TAGS for b in OVER]
    while True:
        progressed = False
        for tag, bench in jt:
            rid=f"gext_{tag}_{bench}"; p=complete(rid, bench, expected)
            if p:
                progressed |= run(rid, "judged", ["/venv/main/bin/python", "-u", "experiments/judge_generations.py", "--generations", str(p.relative_to(ROOT)), "--run-id", f"{rid}_judged", "--num-workers", str(WORKERS)])
        for tag, bench in ot:
            rid=f"gext_{tag}_{bench}"; p=complete(rid, bench, expected)
            if p:
                progressed |= run(rid, "orscore", ["/venv/main/bin/python", "-u", "scripts/external_benches/score_overrefusal.py", "--generations", str(p.relative_to(ROOT)), "--run-id", rid, "--num-workers", str(WORKERS)])
        jd=sum((RESULTS/f"gext_{t}_{b}_judged"/"summary.json").exists() for t,b in jt)
        od=sum((RESULTS/f"gext_{t}_{b}_orscore"/"summary.json").exists() for t,b in ot)
        print(f"[gemma-judge] checkpoint judged={jd}/{len(jt)} orscored={od}/{len(ot)}", flush=True)
        if jd == len(jt) and od == len(ot): return
        if not progressed: time.sleep(15)

if __name__ == "__main__": main()
