#!/usr/bin/env bash
# Train ABL-v7 on Llama-3.2-1B-Instruct.
#
# Run from repo root or anywhere:
#   bash scripts/multimodel_abl_v7/run_llama32_1b_v7.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-python}"

echo "[llama32] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[llama32] PY=${PY}"

exec "$PY" experiments/train_tamper_resistant.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --out outputs/tamper_resistant_llama32_1b_v7.pt \
  --run-id abl_llama32_1b_v7 \
  --train-scope all \
  --abliterate-layers all \
  --attack-ensemble \
  --direction-layer 8 \
  --recompute-direction-every 25 \
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
