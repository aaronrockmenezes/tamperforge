#!/usr/bin/env bash
# Render a training run's events.jsonl into a human-readable log, live.
# The raw tee'd log is unreadable (tqdm rewrites the same line thousands of times);
# events.jsonl is the structured truth. This formats it and keeps appending.
#
#   RUN=results/tamper_resistant_p1b_20260724_223952 bash scripts/watch_train.sh &
#   tail -f results/tamper_resistant_p1b_20260724_223952/train_readable.txt
#
# Overridable: PY(python) EVERY(20s)
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="${RUN:?set RUN=results/<run_id>}"
OUT="$RUN/train_readable.txt"

while true; do
  "${PY:-python}" - "$RUN" > "$OUT.tmp" 2>/dev/null <<'PY'
import json, sys, pathlib
run = pathlib.Path(sys.argv[1])
rows = []
for line in open(run / "events.jsonl"):
    try:
        rows.append(json.loads(line))
    except Exception:
        pass
steps = [r for r in rows if r.get("event") == "step"]
evals = {r["step"]: r for r in rows if r.get("event") == "eval"}
print(f"run: {run.name}    steps logged: {len(steps)}")
print()
print("  step | stage |  loss  | L_task | gib_ce | ref_abl | harm_abl | L_harm |  L_rr  | cleanKL | ifeval")
print("  -----+-------+--------+--------+--------+---------+----------+--------+--------+---------+-------")
for r in steps:
    if r["step"] % 5 and r["step"] not in evals:
        continue
    e = evals.get(r["step"], {})
    ife = e.get("clean_ifeval_acc")
    stage = 1 if r["step"] < 250 else 2
    print(f"  {r['step']:5d} |   {stage}   | {r['loss']:6.2f} | {r['L_task']:6.3f} | "
          f"{r['gib_ce']:6.3f} | {r['ref_abl']:7.3f} | {r.get('harm_abl', float('nan')):8.3f} | "
          f"{r.get('L_harm', float('nan')):6.3f} | {r.get('L_rr', float('nan')):6.4f} | "
          f"{r['L_clean_gen']:7.4f} | {('%.3f' % ife) if ife is not None else '  -  '}")
bad = [r["step"] for r in steps
       if any((v != v or abs(v) == float("inf"))
              for v in (r["loss"], r["gib_ce"], r.get("L_rr", 0.0)))]
if bad:
    print(f"\nNON-FINITE at steps: {bad[:20]}{' ...' if len(bad) > 20 else ''}")
PY
  mv "$OUT.tmp" "$OUT" 2>/dev/null
  sleep "${EVERY:-20}"
done
