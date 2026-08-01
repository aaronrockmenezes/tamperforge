#!/usr/bin/env bash
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

LOG=logs/training_runs/panels_thinking_on.log
echo "=== THINKING-ON panels (qwen-thinking default, dir-layer 20): $(date -u) ===" | tee "$LOG"
echo "=== this is the regime where v8 broke: v11_v8_surg_k16 = 0.448 harmful ===" | tee -a "$LOG"

echo "--- v8 control ---" | tee -a "$LOG"
python -u experiments/version_a_eval_panel.py \
  --checkpoints outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt \
  --n-harmful 64 --max-new-tokens 512 --gen-batch-size 64 --direction-layer 20 \
  --qwen-thinking default --n-direction 256 --n-cap-eval 32 \
  --judge-workers 32 --run-id v8_panel_think 2>&1 | tee -a "$LOG"

echo "--- version_A ---" | tee -a "$LOG"
python -u experiments/version_a_eval_panel.py \
  --checkpoints outputs/version_a_qwen_500.pt.s400.pt \
                outputs/version_a_qwen_500.pt.s450.pt \
                outputs/version_a_qwen_500.pt \
  --n-harmful 64 --max-new-tokens 512 --gen-batch-size 64 --direction-layer 20 \
  --qwen-thinking default --n-direction 256 --n-cap-eval 32 \
  --judge-workers 32 --run-id version_a_panel_think 2>&1 | tee -a "$LOG"

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
