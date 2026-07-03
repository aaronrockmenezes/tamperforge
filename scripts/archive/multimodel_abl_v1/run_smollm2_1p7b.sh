#!/usr/bin/env bash
# Train ABL-v1 on SmolLM2-1.7B-Instruct.
#
# Run from repo root or anywhere:
#   bash scripts/multimodel_abl_v1/run_smollm2_1p7b.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY="${PY:-python}"

echo "[smollm2] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[smollm2] PY=${PY}"

exec "$PY" experiments/train_tamper_resistant.py \
  --model-id HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --out outputs/tamper_resistant_smollm2_1p7b_v1.pt \
  --run-id abl_smollm2_1p7b_v1 \
  --train-scope all \
  --abliterate-layers all \
  --attack-ensemble \
  --direction-layer 12 \
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
