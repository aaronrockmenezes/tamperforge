#!/usr/bin/env bash
# Train ABL-v1 on Qwen3-1.7B. Qwen3 thinking is disabled in the repo's
# chat-template helper when the tokenizer supports enable_thinking=False.
#
# Run from repo root or anywhere:
#   bash scripts/multimodel_abl_v1/run_qwen3_1p7b.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-python}"

echo "[qwen3] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[qwen3] PY=${PY}"

exec "$PY" experiments/train_tamper_resistant.py \
  --model-id Qwen/Qwen3-1.7B \
  --out outputs/tamper_resistant_qwen3_1p7b_v1.pt \
  --run-id abl_qwen3_1p7b_v1 \
  --train-scope all \
  --abliterate-layers all \
  --attack-ensemble \
  --direction-layer 14 \
  --recompute-direction-every 15 \
  --gib-mode argmax \
  --gib-gen-tokens 32 \
  --gib-gen-prompts 2 \
  --lambda-gib 4 \
  --lambda-uncensor 4 \
  --lambda-safe 1 \
  --lambda-reg 0.1 \
  --steps 500 \
  --eval-every 25 \
  --lr 1e-5 \
  --seed 42
