#!/usr/bin/env bash
# Wait for the active rank-1 capability run, export matched held-out
# Heretic-style attacks, then evaluate them with the same compact suite.
set -euo pipefail
cd "$(dirname "$0")/.."

source /venv/main/bin/activate 2>/dev/null || true
if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

WAIT_PID="${WAIT_PID:-}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-15}"
CHECKPOINT="${CHECKPOINT:-outputs/mad_v10_qwen_heretic_20260726_122129.pt.s175.pt}"
MODEL_ID="${MODEL_ID:-outputs/hf_qwen/Qwen3-0.6B}"
RUN_ID="${RUN_ID:-mad_v10_s175_heretic_style_caps_$(date +%Y%m%d_%H%M%S)}"
EXPORT_ROOT="${EXPORT_ROOT:-outputs/${RUN_ID}_vllm}"

if [ -n "${WAIT_PID}" ]; then
  echo "[queue] waiting for rank-1 eval pid=${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep "${WAIT_POLL_SECONDS}"
  done
fi

echo "[queue] exporting matched Heretic-style variants"
python scripts/export_mad_v10_heretic_style_variants.py \
  --checkpoint "${CHECKPOINT}" \
  --model-id "${MODEL_ID}" \
  --out-root "${EXPORT_ROOT}" \
  --attack-layers 10-27 \
  --alpha-min 0.2 \
  --alpha-max 0.6 \
  --attack-seed 20260726 \
  --direction-seed 20260928

echo "[queue] evaluating trained and base Heretic-style attacks"
RUN_ID="${RUN_ID}" \
EXPORT_ROOT="${EXPORT_ROOT}" \
VARIANTS="trained_heretic_attacked,base_heretic_attacked" \
MAX_MODEL_LEN=8192 \
MAX_GEN_TOKS=4096 \
MAX_NUM_SEQS=256 \
EVAL_LIMIT=200 \
MMLU_LIMIT_PER_TASK=40 \
AGI_TASK=agieval_lsat_ar \
GPU_MEM=0.90 \
bash scripts/run_mad_v10_s175_vllm_caps.sh

echo "[queue] done: results/${RUN_ID}"
