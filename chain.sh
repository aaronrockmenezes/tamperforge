#!/usr/bin/env bash
# AFK chain: wait for the direction-layer sweep, smoke version_B, then train it 500 steps.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
CHAIN=logs/training_runs/chain.log
echo "=== chain start $(date -u) ===" | tee "$CHAIN"

echo "[1/3] waiting for dl_sweep..." | tee -a "$CHAIN"
until grep -aq "=== DONE" logs/training_runs/dl_sweep.log 2>/dev/null; do sleep 30; done
echo "[1/3] dl_sweep done $(date -u)" | tee -a "$CHAIN"

echo "[2/3] version_B smoke (6 steps)" | tee -a "$CHAIN"
python -u experiments/train_tamper_resistant_v8.py \
  --model-id Qwen/Qwen3-0.6B --out outputs/vb_smoke.pt \
  --train-scope all --abliterate-layers all --attack-ensemble \
  --attack-profile version_b --attack-layers 10-27 --direction-layer 20 --no-grad-checkpoint \
  --gib-mode argmax --gib-gen-tokens 8 --gib-gen-prompts 1 \
  --lambda-gib 8 --lambda-uncensor 4 --lambda-safe 1 --lambda-reg 0.1 --lambda-clean 3 \
  --clean-gen-prompts 1 --clean-gen-tokens 8 --n-direction 32 --version-a-n-cap 32 \
  --steps 6 --eval-every 1000 --save-every 1000 --lr 1e-5 --seed 42 \
  > logs/training_runs/vb_smoke.log 2>&1
if [ $? -ne 0 ]; then
  echo "[2/3] SMOKE FAILED -- not launching training. tail:" | tee -a "$CHAIN"
  tail -25 logs/training_runs/vb_smoke.log | tee -a "$CHAIN"
  exit 1
fi
rm -f outputs/vb_smoke.pt
echo "[2/3] smoke OK $(date -u)" | tee -a "$CHAIN"

echo "[3/3] version_B 500-step training" | tee -a "$CHAIN"
RUN=version_b_qwen_500
python -u experiments/train_tamper_resistant_v8.py \
  --model-id Qwen/Qwen3-0.6B --out outputs/${RUN}.pt \
  --train-scope all --abliterate-layers all --attack-ensemble \
  --attack-profile version_b --attack-layers 10-27 --direction-layer 20 \
  --no-grad-checkpoint \
  --recompute-direction-every 25 --gib-mode argmax --gib-gen-tokens 32 --gib-gen-prompts 2 \
  --lambda-gib 8 --lambda-uncensor 4 --lambda-safe 1 --lambda-reg 0.1 \
  --lambda-clean 3 --clean-gen-prompts 2 --clean-gen-tokens 32 \
  --clean-start-step 250 --clean-ramp-steps 100 \
  --stage2-lambda-gib 8 --stage2-lambda-safe 4 \
  --ifeval-in-loop --ifeval-probe-n 24 \
  --n-direction 256 --version-a-n-cap 256 \
  --steps 500 --eval-every 25 --save-every 25 --lr 1e-5 --seed 42 \
  2>&1 | tee logs/training_runs/${RUN}.log
echo "=== chain DONE $(date -u) ===" | tee -a "$CHAIN"
df -h /workspace | tail -1 | tee -a "$CHAIN"
