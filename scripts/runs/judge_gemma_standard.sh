#!/usr/bin/env bash
# Judge already-generated Gemma standard artifacts only. No generation/model load.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

WORKERS="${JUDGE_WORKERS:-96}"
LOG=logs/eval/gemma_standard_judge_$(date -u +%Y%m%dT%H%M%S).log
mkdir -p logs/eval results
rm -f results/.defer_api_scoring

TAGS=(vgg_clean vgg_rank1 vgg_surg_k16 vgg_her_s0)
judge_one () {
  local rid="$1" expected="$2" src="results/$1/generations.jsonl"
  local out="results/${rid}_judged/summary.json" got
  if [ -s "$out" ]; then
    echo "[skip] $rid already judged" | tee -a "$LOG"
    return
  fi
  [ -s "$src" ] || { echo "[FAIL] missing $src" | tee -a "$LOG"; exit 1; }
  got=$(wc -l < "$src" | tr -d ' ')
  [ "$got" = "$expected" ] || { echo "[FAIL] $rid rows=$got expected=$expected" | tee -a "$LOG"; exit 1; }
  echo "[judge] $rid rows=$got workers=$WORKERS" | tee -a "$LOG"
  python -u experiments/judge_generations.py --generations "$src" \
    --run-id "${rid}_judged" --num-workers "$WORKERS" >>"$LOG" 2>&1
  [ -s "$out" ] || { echo "[FAIL] no summary for $rid" | tee -a "$LOG"; exit 1; }
}

for tag in "${TAGS[@]}"; do
  judge_one "$tag" 520
  judge_one "${tag}_xssafe" 250
  judge_one "${tag}_xsunsafe" 200
done

MT_TAGS=(mtb_gbase_clean mtb_vgg mtb_vgg_her_s0)
for tag in "${MT_TAGS[@]}"; do
  [ -s "results/$tag/generations.jsonl" ] || { echo "[FAIL] missing MT-Bench $tag" | tee -a "$LOG"; exit 1; }
done
echo "[judge] MT-Bench single scores" | tee -a "$LOG"
python -u experiments/mtbench_single.py --repeats 3 --num-workers 32 \
  --tags "${MT_TAGS[@]}" >>"$LOG" 2>&1
python -u experiments/mtbench_pairwise.py --a mtb_vgg --b mtb_gbase_clean \
  --label-a vgg --label-b gbase_clean >>"$LOG" 2>&1
python -u experiments/mtbench_pairwise.py --a mtb_vgg_her_s0 --b mtb_gbase_clean \
  --label-a vgg_her_s0 --label-b gbase_clean >>"$LOG" 2>&1

python - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
R=Path("results")
tags=["vgg_clean","vgg_rank1","vgg_surg_k16","vgg_her_s0"]
for tag in tags:
    for suffix, expected in [("",520),("_xssafe",250),("_xsunsafe",200)]:
        rid=tag+suffix
        p=R/f"{rid}_judged"/"summary.json"
        d=json.loads(p.read_text())
        print(rid, f"n={d.get('n')}/{expected}", f"parse_failures={d.get('parse_failures')}",
              f"harm={d.get('harmful_actionable_rate')}")
print("STANDARD_JUDGING_COMPLETE 12/12")
PY
