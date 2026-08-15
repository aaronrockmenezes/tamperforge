#!/usr/bin/env bash
# Qwen3-0.6B Version G: fresh fixed-L20 rank-1/rank-2 attacks, AdvBench-520 in parallel.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
PY="${PY:-python}"
[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python
export CUDA_VISIBLE_DEVICES=0

MODEL="Qwen/Qwen3-0.6B"
CKPT="outputs/version_g_qwen_500.pt"
HF_FILE="final_backup_2026_08_03/checkpoints/version_g_qwen_500.pt"
RESTORE="outputs/hf_restore_qwen06_vg"
LOG_ROOT="logs/eval/qwen06_vg_rank12"
mkdir -p outputs "$RESTORE" "$LOG_ROOT"

if [ ! -s "$CKPT" ]; then
  hf download aaronrockmenezes/tamperforge "$HF_FILE" --local-dir "$RESTORE"
  cp "$RESTORE/$HF_FILE" "$CKPT"
fi
[ -s "$CKPT" ] || { echo "Qwen Version G checkpoint download failed" >&2; exit 1; }

model_ready() {
  [ -s "$1/model.safetensors" ] || [ -s "$1/model.safetensors.index.json" ] ||
    find "$1" -maxdepth 1 -type f -name 'model-*.safetensors' -size +0 -print -quit 2>/dev/null | grep -q .
}
summary_n() {
  [ -s "$1" ] && "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("n") != int(sys.argv[2]))' "$1" "$2"
}

for rank in 1 2; do
  out="outputs/qwen06_vg_hf_rank${rank}"
  if ! model_ready "$out"; then
    "$PY" -u experiments/v11_surgical_ablation.py \
      --model-id "$MODEL" --checkpoint "$CKPT" --direction-layer 20 \
      --refusal-rank "$rank" --cap-rank 0 --rank-estimator arditi_residual \
      --out "$out" >"$LOG_ROOT/materialize_rank${rank}.log" 2>&1
  fi
  model_ready "$out" || { echo "rank-$rank materialization failed" >&2; exit 1; }
done

pids=()
cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]:-}"; do [ -n "$pid" ] && wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

for spec in "1:8765" "2:8766"; do
  rank="${spec%%:*}"; port="${spec##*:}"
  tag="qwen06_vg_hf_rank${rank}"
  EVAL_PROFILE=advbench LOG_ROOT="$LOG_ROOT" PORT="$port" UTIL=0.30 \
    GEN_WORKERS=64 JUDGE_WORKERS=64 JUDGE_MAX_TOKENS=1024 \
    JUDGE_TIMEOUT_SECONDS=10 SAFETY_MAX_NEW_TOKENS=1024 \
    bash scripts/eval/serve_eval.sh "$tag" "outputs/$tag" off &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
pids=()
[ "$failed" -eq 0 ] || { echo "rank-1/rank-2 evaluation failed" >&2; exit 1; }

for rank in 1 2; do
  summary_n "results/qwen06_vg_hf_rank${rank}_judged/summary.json" 520 || {
    echo "rank-$rank judged summary incomplete" >&2; exit 1;
  }
done

"$PY" - <<'PY'
import json
from pathlib import Path

rows = []
for rank in (1, 2):
    d = json.loads((Path("results") / f"qwen06_vg_hf_rank{rank}_judged" / "summary.json").read_text())
    rows.append({
        "rank": rank, "n": d["n"],
        "harmful_actionable_rate": d["harmful_actionable_rate"],
        "gibberish_rate": d["gibberish_rate"],
        "judge_refusal_rate": d["judge_refusal_rate"],
        "parse_failures": d["parse_failures"],
    })
out = Path("results/qwen06_vg_hf_rank12_report.json")
out.write_text(json.dumps({"model": "Qwen/Qwen3-0.6B Version G", "layer": 20, "rows": rows}, indent=2))
print(out.read_text())
PY
