#!/usr/bin/env python
"""Live monitor for the training run happening right now.

    python run_monitor.py                # snapshot
    python run_monitor.py -w             # refresh every 30s until the run ends
    python run_monitor.py -w -i 10       # ...every 10s
    python run_monitor.py --run results/tamper_resistant_p1b_20260801_225415

Complements va_status.py, which prints ONE ROW PER STEP joined to its attack. This answers
"is the run healthy and how far along is it" -- trend, not detail.

The three things that actually matter while a run is in flight:
  1. gib_ce vs gap_target. L_gib = relu(gap - gib_ce), so once gib_ce > gap the objective is
     SATISFIED and contributes no gradient. If it sits satisfied from early on, the run is
     not learning the wall -- it already thinks it is done.
  2. refuse_abl (ref_abl). The attacked model's refusal margin. This is the wall forming.
  3. The attack mix actually drawn, not the mix configured. version_B samples write-only
     ~5% of steps; if the realised share is far off, the sampler is not doing what you think.
"""
from __future__ import annotations
import argparse, glob, json, os, re, statistics as st, subprocess, sys, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # repo root, not scripts/tools/


def newest_run() -> str | None:
    ds = glob.glob(str(ROOT / "results" / "tamper_resistant_p1b_*"))
    return max(ds, key=os.path.getmtime) if ds else None


def newest_log() -> Path | None:
    ls = glob.glob(str(ROOT / "logs" / "**" / "*.log"), recursive=True)
    ls = [l for l in ls if "p1b-A steps" in _tail(l, 3000)]
    return Path(max(ls, key=os.path.getmtime)) if ls else None


def _tail(p, n=4000) -> str:
    try:
        return Path(p).read_bytes()[-n:].decode("utf-8", "replace").replace("\r", "\n")
    except OSError:
        return ""


def spark(vals, lo=None, hi=None, width=32) -> str:
    if not vals: return ""
    v = vals[-width:]
    lo = min(v) if lo is None else lo
    hi = max(v) if hi is None else hi
    if hi - lo < 1e-9: return "─" * len(v)
    bars = "▁▂▃▄▅▆▇█"
    return "".join(bars[min(7, int((x - lo) / (hi - lo) * 7.99))] for x in v)


def show(run: str, log: Path | None, gap: float) -> bool:
    rows = [json.loads(l) for l in open(f"{run}/events.jsonl") if l.strip()]
    rows = [r for r in rows if "step" in r]
    if not rows:
        print("no steps logged yet"); return True
    last = rows[-1]
    tail = _tail(log, 6000) if log else ""
    bar = [l for l in tail.split("\n") if "p1b-A steps" in l]
    done = "TRAIN_RC" in tail
    print("\033[2J\033[H", end="")
    print(f"\033[1mrun\033[0m {Path(run).name}   steps logged {len(rows)}"
          + ("   \033[1mFINISHED\033[0m" if done else ""))
    if bar: print("     " + re.sub(r"\s+", " ", bar[-1]).strip()[-76:])

    g = [r["gib_ce"] for r in rows if r.get("gib_ce") is not None]
    ra = [r["ref_abl"] for r in rows if r.get("ref_abl") is not None]
    ls = [r["loss"] for r in rows if r.get("loss") is not None]
    sat = sum(1 for x in g[-50:] if x >= gap)
    print(f"\n\033[1mobjective\033[0m (gap_target {gap})")
    print(f"  gib_ce   {g[-1]:7.3f}  med50 {st.median(g[-50:]):6.3f}  {spark(g)}")
    # WHEN it became satisfied is the diagnostic: satisfied from the start means L_gib never
    # trained anything; satisfied later means the wall formed and then stopped being pushed.
    gsteps = [(r["step"], r["gib_ce"]) for r in rows if r.get("gib_ce") is not None]
    first_sat = next((st_ for st_, v in gsteps if v >= gap), None)
    unsat_after = [st_ for st_, v in gsteps if v < gap]
    frac = sum(1 for _, v in gsteps if v >= gap) / max(len(gsteps), 1)
    print(f"           satisfied {sat}/50 recent, {frac:.0%} overall; "
          f"first satisfied at step {first_sat}"
          + (f", last UNsatisfied at {max(unsat_after)}" if unsat_after else ""))
    if sat > 40:
        print("           -> L_gib contributes NO gradient right now (relu margin met)")
    if ra: print(f"  ref_abl  {ra[-1]:7.3f}  med50 {st.median(ra[-50:]):6.3f}  {spark(ra)}")
    if ls: print(f"  loss     {ls[-1]:7.3f}  med50 {st.median(ls[-50:]):6.3f}  {spark(ls)}")

    # the newest row may be an event without loss fields; walk back to the last one that has them
    lc = next((r for r in reversed(rows) if r.get("W_task") is not None), {})
    print(f"\n\033[1mloss components\033[0m (weighted, step {lc.get('step','?')})")
    comp = [(k[2:], lc.get(k)) for k in
            ("W_task","W_safe","W_gib","W_clean_gen","W_reg","W_uncensor","W_harm")
            if lc.get(k) is not None]
    print("  " + ("  ".join(f"{n}={v:.3f}" for n, v in comp) if comp else "(none logged)"))

    print("\n\033[1mattack mix drawn\033[0m (realised, not configured)")
    fam = Counter("canonical" if "canonical" in (r.get("attack_tag") or "") else
                  ("surgical" if r.get("attack_variant") == "surgical" else "other")
                  for r in rows)
    tot = sum(fam.values())
    wo = sum(1 for r in rows if r.get("attack_write_only"))
    pl = sum(1 for r in rows if r.get("attack_per_layer"))
    print("  " + "  ".join(f"{k} {v/tot:.3f}" for k, v in fam.most_common()))
    print(f"  write_only {wo/tot:.3f}   per_layer {pl/tot:.3f}   "
          f"n_matrices med {st.median([r.get('attack_n_matrices',0) for r in rows]):.0f}")

    prev = [l for l in tail.split("\n") if "advbench-preview attacked" in l]
    if prev:
        print("\n\033[1mlatest attacked preview\033[0m")
        print("  " + re.sub(r"\s+", " ", prev[-1]).strip()[:150])
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("-w", "--watch", action="store_true")
    ap.add_argument("-i", "--interval", type=int, default=30)
    ap.add_argument("--gap", type=float, default=4.0, help="gap_target the run was launched with")
    a = ap.parse_args()
    run = a.run or newest_run()
    if not run:
        sys.exit("no run dir under results/tamper_resistant_p1b_*")
    log = newest_log()
    while True:
        finished = show(run, log, a.gap)
        if not a.watch or finished:
            break
        time.sleep(a.interval)
