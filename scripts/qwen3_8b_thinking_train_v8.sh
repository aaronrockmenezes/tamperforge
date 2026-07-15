#!/usr/bin/env bash
# Train Qwen3-8B ABL-v8 in thinking mode on a 96GB RTX PRO 6000.
#
# Required:
#   DL=<best sweep layer> bash scripts/qwen3_8b_thinking_train_v8.sh
#
# Defaults are full precision/full optimizer-state AdamW, not adamw8bit.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DL="${DL:?set DL from qwen3_8b_thinking_dl_sweep.sh}"
OUT="${OUT:-outputs/tamper_resistant_qwen3_8b_thinking_v8.pt}"
TRAIN_SCOPE="${TRAIN_SCOPE:-all}"
STEPS="${STEPS:-500}"
SAVE_EVERY="${SAVE_EVERY:-25}"
SEED="${SEED:-42}"
OPTIM="${OPTIM:-adamw}"

export TF_QWEN_THINKING=on
export TF_IFEVAL_MAX_NEW="${TF_IFEVAL_MAX_NEW:-5000}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PY" experiments/train_tamper_resistant_v8.py \
  --model-id "$MODEL" \
  --out "$OUT" \
  --run-id "train_qwen3_8b_thinking_v8_s${SEED}" \
  --qwen-thinking on \
  --train-scope "$TRAIN_SCOPE" \
  --abliterate-layers all \
  --attack-ensemble \
  --direction-layer "$DL" \
  --recompute-direction-every 25 \
  --gib-mode argmax \
  --gib-gen-tokens "${GIB_GEN_TOKENS:-128}" \
  --gib-gen-prompts "${GIB_GEN_PROMPTS:-1}" \
  --lambda-gib "${LGIB:-8}" \
  --lambda-uncensor 4 \
  --lambda-safe 1 \
  --lambda-reg 0.1 \
  --lambda-clean "${LCLEAN:-3}" \
  --clean-gen-prompts "${CLEAN_GEN_PROMPTS:-1}" \
  --clean-gen-tokens "${CLEAN_GEN_TOKENS:-128}" \
  --clean-start-step "${CLEAN_START_STEP:-250}" \
  --clean-ramp-steps "${CLEAN_RAMP_STEPS:-100}" \
  --stage2-lambda-gib "${S2GIB:-4}" \
  --stage2-lambda-safe "${S2SAFE:-4}" \
  --ifeval-in-loop \
  --ifeval-probe-n "${IFEVAL_PROBE_N:-12}" \
  --ifeval-max-new "${IFEVAL_MAX_NEW:-5000}" \
  --save-every "$SAVE_EVERY" \
  --steps "$STEPS" \
  --eval-every "${EVAL_EVERY:-25}" \
  --task-batch "${TASK_BATCH:-1}" \
  --refusal-batch "${REFUSAL_BATCH:-1}" \
  --lr "${LR:-1e-5}" \
  --optim "$OPTIM" \
  --seed "$SEED"

echo "### trained -> ${OUT} (+ snapshots). Next:"
echo "DL=${DL} STEM=${OUT} bash scripts/qwen3_8b_thinking_pick_v8.sh"
