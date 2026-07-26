#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [ -z "${PYTHON_BIN+x}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif [ -x "/venv/main/bin/python" ]; then
    PYTHON_BIN="/venv/main/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [ -z "${MODEL_ID+x}" ]; then
  if [ -d "outputs/hf_qwen/Qwen3-0.6B" ]; then
    MODEL_ID="outputs/hf_qwen/Qwen3-0.6B"
  elif [ -d "outputs/hf_qwen" ]; then
    CACHED_MODEL="$(find outputs/hf_qwen -path '*Qwen3-0.6B*' -name config.json -print -quit)"
    if [ -n "$CACHED_MODEL" ]; then
      MODEL_ID="$(dirname "$CACHED_MODEL")"
    else
      MODEL_ID="Qwen/Qwen3-0.6B"
    fi
  else
    MODEL_ID="Qwen/Qwen3-0.6B"
  fi
fi

RUN_ID="${RUN_ID:-mad_v10_qwen_$(date +%Y%m%d_%H%M%S)}"
LOG_PATH="logs/${RUN_ID}.log"
EXTRA_ARGS=()

if [ -n "${UNSAFE_JSONL:-}" ]; then
  EXTRA_ARGS+=(--unsafe-jsonl "$UNSAFE_JSONL")
fi

PYTHONPATH=src "$PYTHON_BIN" experiments/train_mad_v10.py \
  --model-id "$MODEL_ID" \
  --out "outputs/${RUN_ID}.pt" \
  --run-id "$RUN_ID" \
  --qwen-thinking off \
  --attack-profile mixed \
  --attack-layers 12-27 \
  --train-scope all \
  --cap-datasets gsm8k,arc,tiny_if \
  --steps "${STEPS:-200}" \
  --eval-every "${EVAL_EVERY:-25}" \
  --print-every "${PRINT_EVERY:-1}" \
  --save-every "${SAVE_EVERY:-25}" \
  --cap-batch "${CAP_BATCH:-2}" \
  --lambda-clean-cap "${LAMBDA_CLEAN_CAP:-1.0}" \
  --lambda-clean-kl "${LAMBDA_CLEAN_KL:-0.1}" \
  --lambda-clean-safe "${LAMBDA_CLEAN_SAFE:-0.5}" \
  --lambda-cap "${LAMBDA_CAP:-1.0}" \
  --lambda-div "${LAMBDA_DIV:-0.25}" \
  --lambda-bad-ul "${LAMBDA_BAD_UL:-0.5}" \
  --lambda-reg "${LAMBDA_REG:-0.01}" \
  --cap-margin "${CAP_MARGIN:-2.0}" \
  --div-margin "${DIV_MARGIN:-2.0}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_PATH"
