#!/usr/bin/env python
"""One-screen status for whatever tamperforge is doing right now.

    python tf_status.py            # everything
    python tf_status.py --runs     # training runs only
    python tf_status.py --grep vbl # filter jobs/logs by substring

Reads tmux, nvidia-smi, disk, the newest training log, and every judged summary.
No arguments needed and nothing is written -- safe to run against a live box.
"""
from __future__ import annotations
import argparse, glob, json, os, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root, not scripts/tools/
LOGS = ROOT / "logs"
RES = ROOT / "results"
# reference points so a number means something without opening the handoff
REF = {"qwen": {"clean": 0.2577, "ceiling": 0.6596},
       "llama": {"clean": 0.0019, "ceiling": 0.8269}}


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def hdr(t: str) -> None:
    print(f"\n\033[1m{t}\033[0m\n" + "-" * len(t))


def jobs(filt: str | None) -> None:
    hdr("running")
    out = sh("tmux ls 2>/dev/null")
    rows = [l for l in out.splitlines() if l and (not filt or filt in l)]
    if not rows:
        print("  nothing in tmux")
    for l in rows:
        print("  " + l)
    gpu = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader")
    procs = sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    print(f"  gpu: {gpu}")
    # NOTE: on vast.ai containers nvidia-smi reports HOST pids, which never exist in the
    # container's /proc. Do NOT infer "orphaned/leaked" from a missing /proc entry -- that
    # check flags every live process, including the training run you just started.
    # A genuine leak looks like: memory held while tmux is empty AND it does not drop after
    # a few minutes.
    for p in procs.splitlines():
        print(f"  gpu proc {p} (host pid; not resolvable in-container)")
    print("  disk: " + sh("df -h /workspace | tail -1 | awk '{print $3\" used, \"$4\" free\"}'"))


def progress(filt: str | None) -> None:
    hdr("training progress")
    cands = sorted(glob.glob(str(LOGS / "**" / "*_500.log"), recursive=True) +
                   glob.glob(str(LOGS / "**" / "*smoke*.log"), recursive=True),
                   key=os.path.getmtime, reverse=True)
    cands = [c for c in cands if not filt or filt in c]
    if not cands:
        print("  no training logs")
        return
    for c in cands[:3]:
        raw = Path(c).read_bytes()[-4000:].decode("utf-8", "replace").replace("\r", "\n")
        bar = [l for l in raw.split("\n") if "p1b-A steps" in l]
        age = (time.time() - os.path.getmtime(c)) / 60
        state = "LIVE" if age < 3 else f"idle {age:.0f}m"
        line = re.sub(r"\s+", " ", bar[-1]).strip()[-70:] if bar else "(no progress line)"
        # tqdm's s/it is cumulative, so an eval_every cycle inflates it and the ETA swings.
        # Report the raw line; do not quote a single ETA as if it were stable.
        done = "TRAIN_RC=0" in raw
        print(f"  [{state}{' DONE' if done else ''}] {Path(c).name}")
        print(f"      {line}")


def results(filt: str | None) -> None:
    hdr("judged results (newest 14)")
    js = sorted(glob.glob(str(RES / "*_judged" / "summary.json")), key=os.path.getmtime, reverse=True)
    js = [j for j in js if not filt or filt in j]
    if not js:
        print("  none")
        return
    print("  %-26s %6s %8s %8s %8s  %s" % ("run", "n", "harm", "gib", "ref", "headroom"))
    for j in js[:14]:
        try:
            s = json.load(open(j))
            u, n = s["usefulness_counts"], s["n"]
        except Exception:
            continue
        h = u.get("harmful_actionable", 0) / n
        name = Path(j).parent.name.replace("_judged", "")
        # pick the reference frame from the run-id prefix; llama runs start with l/hl
        r = REF["llama"] if name.startswith(("lbase", "hlbase", "lvb")) else REF["qwen"]
        hr = 100 * (h - r["clean"]) / (r["ceiling"] - r["clean"])
        print("  %-26s %6d %8.4f %8.4f %8.4f  %+7.1f%%" % (
            name, n, h, u.get("gibberish", 0) / n, u.get("refused", 0) / n, hr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="store_true", help="training progress only")
    ap.add_argument("--grep", default=None, help="filter by substring")
    a = ap.parse_args()
    if a.runs:
        progress(a.grep)
    else:
        jobs(a.grep); progress(a.grep); results(a.grep)
    print()
