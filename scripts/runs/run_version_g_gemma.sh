#!/usr/bin/env bash
# Version G replication on Gemma 3 1B. DL=14 is Gemma's measured base-ablation
# sweep peak in this repo; the Version G recipe otherwise matches Qwen/Llama.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs logs/probes outputs

TAG=version_g_gemma_500
MODEL_ID=google/gemma-3-1b-it
REF=data/extended_refusals_advbench.json
HARM=data/harm_targets_qwen.json
LOG=logs/training_runs/${TAG}.log

say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
[ -s "$REF" ] || { say "[FAIL] missing $REF"; exit 1; }
[ -s "$HARM" ] || { say "[FAIL] missing $HARM"; exit 1; }

if [ -s "outputs/${TAG}.pt" ]; then
  say "[skip] checkpoint already exists"
else
  say "=== TRAIN $TAG ==="
  TF_ATTN_IMPL=eager python -u experiments/train_tamper_resistant_v8.py \
    --model-id "$MODEL_ID" --out "outputs/${TAG}.pt" \
    --train-scope all --abliterate-layers all --attack-ensemble \
    --attack-profile version_b --attack-layers all --direction-layer 14 \
    --version-a-p-canonical 0.10 --version-b-p-heretic 0.35 \
    --no-grad-checkpoint --recompute-direction-every 25 \
    --lambda-rr 4 --harm-targets "$HARM" --rr-layers last_half \
    --lambda-gib 0 --stage2-lambda-gib 0 \
    --lambda-uncensor 4 --uncensor-margin 4 \
    --lambda-harm 4 --harm-margin 4 \
    --lambda-safe 4 --stage2-lambda-safe 4 \
    --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 64 \
    --clean-start-step 0 --clean-ramp-steps 100 \
    --refusal-file "$REF" --refusal-max-len 384 \
    --gsm8k-probe-n 8 --gsm8k-probe-max-new 256 \
    --n-direction 256 --version-a-n-cap 256 \
    --steps 500 --eval-every 50 --save-every 500 --lr 1e-5 --seed 42 \
    --qwen-thinking off 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  say "trainer_rc=$rc"
  [ "$rc" = 0 ] || exit "$rc"
fi

[ -s "outputs/${TAG}.pt" ] || { say "[FAIL] missing checkpoint"; exit 1; }
say "checkpoint=$(stat -c %s outputs/${TAG}.pt) bytes"
say "=== TRAINING COMPLETE ==="
