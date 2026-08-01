#!/usr/bin/env bash
# version_A 500-step run. Launched under tmux session "va"; log tees to logs/training_runs/.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

RUN=version_a_qwen_500
LOG=logs/training_runs/${RUN}.log
mkdir -p logs/training_runs outputs

echo "=== version_A 500-step run: $(date -u) ===" | tee "$LOG"
echo "=== log: $LOG ===" | tee -a "$LOG"

python -u experiments/train_tamper_resistant_v8.py \
  --model-id Qwen/Qwen3-0.6B --out outputs/${RUN}.pt \
  --train-scope all --abliterate-layers all --attack-ensemble \
  --attack-profile version_a --attack-layers 10-27 --direction-layer 20 \
  --no-grad-checkpoint \
  --recompute-direction-every 25 --gib-mode argmax --gib-gen-tokens 32 --gib-gen-prompts 2 \
  --lambda-gib 8 --lambda-uncensor 4 --lambda-safe 1 --lambda-reg 0.1 \
  --lambda-clean 3 --clean-gen-prompts 2 --clean-gen-tokens 32 \
  --clean-start-step 250 --clean-ramp-steps 100 \
  --stage2-lambda-gib 8 --stage2-lambda-safe 4 \
  --ifeval-in-loop --ifeval-probe-n 24 \
  --n-direction 256 --version-a-n-cap 256 \
  --steps 500 --eval-every 25 --save-every 25 \
  --lr 1e-5 --seed 42 2>&1 | tee -a "$LOG"

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"
ls -lh outputs/${RUN}*.pt | tee -a "$LOG"
