#!/usr/bin/env bash
# Reproduce gemma-3-1b ABL-v7 at a NEW SEED (multi-seed rigor: is ABL-v7 a lucky run?).
# EXACT canonical v7 recipe from results/p1b_a_ensemble_v7/manifest.json — do not drift.
# The original product is seed 42; this trains seed 1/2 for an n=3 headline.
#
#   SEED=1 CUDA_VISIBLE_DEVICES=2 bash scripts/multimodel_abl_v7/run_gemma3_1b_v7_seed.sh
#   SEED=2 CUDA_VISIBLE_DEVICES=3 bash scripts/multimodel_abl_v7/run_gemma3_1b_v7_seed.sh

set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
PY="${PY:-python}"
SEED="${SEED:?set SEED=1 or SEED=2}"

echo "[gemma-v7-seed$SEED] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} PY=${PY}"

exec "$PY" experiments/train_tamper_resistant.py \
  --model-id google/gemma-3-1b-it \
  --out outputs/tamper_resistant_p1b_v7_seed${SEED}.pt \
  --run-id abl_gemma3_1b_v7_seed${SEED} \
  --train-scope all \
  --abliterate-layers all \
  --attack-ensemble \
  --direction-layer 13 \
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
  --seed "${SEED}"
