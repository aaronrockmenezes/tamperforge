#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ID="${RUN_ID:-mad_v8_qwen_rank1_heretic_gib8}"
MODEL_ID="${MODEL_ID:-outputs/hf_qwen/Qwen3-0.6B}"
OUT="${OUT:-outputs/${RUN_ID}.pt}"

"${PY:-/venv/main/bin/python}" experiments/train_tamper_resistant_v8.py \
  --model-id "$MODEL_ID" \
  --out "$OUT" \
  --run-id "$RUN_ID" \
  --qwen-thinking off \
  --train-scope all \
  --abliterate-layers all \
  --attack-ensemble \
  --attack-profile rank1_heretic_mix \
  --attack-write-scope all_write \
  --attack-layers 10-27 \
  --attack-alpha-min 0.2 \
  --attack-alpha-max 0.8 \
  --direction-layer 13 \
  --n-direction 256 \
  --recompute-direction-every 25 \
  --gib-mode argmax \
  --gib-gen-tokens 32 \
  --gib-gen-prompts 2 \
  --gap-target 8 \
  --lambda-gib 8 \
  --lambda-uncensor 4 \
  --uncensor-margin 4 \
  --lambda-safe 1 \
  --lambda-reg 0.1 \
  --lambda-clean 3 \
  --clean-gen-prompts 2 \
  --clean-gen-tokens 32 \
  --clean-start-step 250 \
  --clean-ramp-steps 100 \
  --stage2-lambda-gib 4 \
  --stage2-lambda-safe 4 \
  --ifeval-in-loop \
  --ifeval-probe-n 24 \
  --advbench-preview-tokens 50 \
  --save-every 50 \
  --steps 500 \
  --eval-every 25 \
  --lr 1e-5 \
  --seed 42
