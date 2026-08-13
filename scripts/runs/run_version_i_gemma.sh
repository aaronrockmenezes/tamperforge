#!/usr/bin/env bash
# Version I: force safety to survive deletion of a rank-8 refusal subspace.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs logs/eval logs/heretic outputs results

TAG=version_i_gemma_r8_500
LOG=logs/training_runs/${TAG}.log

if [ ! -s "outputs/${TAG}.pt" ]; then
  TF_ATTN_IMPL=eager python -u experiments/train_tamper_resistant_v8.py \
    --model-id google/gemma-3-1b-it --out "outputs/${TAG}.pt" \
    --run-id "${TAG}_train" \
    --train-scope all --abliterate-layers all \
    --direction-layer 14 --attack-rank 8 --recompute-direction-every 25 \
    --lambda-attacked-safe 4 --lambda-attacked-benign 3 \
    --lambda-safe 4 --stage2-lambda-safe 4 \
    --lambda-uncensor 0 --lambda-harm 0 --lambda-rr 0 \
    --lambda-gib 0 --stage2-lambda-gib 0 --lambda-shutdown 0 \
    --lambda-clean 3 --lambda-reg 0.1 \
    --clean-gen-prompts 2 --clean-gen-tokens 64 \
    --clean-start-step 0 --clean-ramp-steps 100 \
    --refusal-file data/extended_refusals_advbench.json --refusal-max-len 384 \
    --gsm8k-probe-n 8 --gsm8k-probe-max-new 256 \
    --n-direction 256 --steps 500 --eval-every 50 --save-every 500 \
    --lr 1e-5 --seed 42 --no-grad-checkpoint --qwen-thinking off \
    2>&1 | tee "$LOG"
fi

[ -s "outputs/${TAG}.pt" ] || { echo "[FAIL] missing outputs/${TAG}.pt"; exit 1; }

TAG="$TAG" SHORT=vig WAIT_ON=none \
  MODEL_ID=google/gemma-3-1b-it DIRECTION_LAYER=14 \
  BASE_TAG=gbase_clean BASE_HF=google/gemma-3-1b-it \
  ENFORCE_GATES=0 JUDGE_WORKERS=96 SKIP_MMLU=1 \
  bash scripts/runs/chain_f.sh
