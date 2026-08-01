#!/usr/bin/env bash
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/panels_walledai.log
echo "=== WALLEDAI-sourced panels: $(date -u) ===" | tee "$LOG"
echo "=== validation target: v8 surgical_k16 thinking-on should approach 0.448 ===" | tee -a "$LOG"

# v8 thinking-on FIRST -- it is the validation number; if it does not reproduce, stop.
echo "--- v8 thinking-ON (validation) ---" | tee -a "$LOG"
python -u experiments/version_a_eval_panel.py \
  --checkpoints outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt \
  --n-harmful 64 --max-new-tokens 512 --gen-batch-size 64 --direction-layer 20 \
  --advbench-source walledai --qwen-thinking default --n-direction 256 --n-cap-eval 32 \
  --judge-workers 32 --run-id v8_wall_think 2>&1 | tee -a "$LOG"

echo "--- version_A thinking-ON ---" | tee -a "$LOG"
python -u experiments/version_a_eval_panel.py \
  --checkpoints outputs/version_a_qwen_500.pt.s400.pt \
                outputs/version_a_qwen_500.pt.s450.pt \
                outputs/version_a_qwen_500.pt \
  --n-harmful 64 --max-new-tokens 512 --gen-batch-size 64 --direction-layer 20 \
  --advbench-source walledai --qwen-thinking default --n-direction 256 --n-cap-eval 32 \
  --judge-workers 32 --run-id version_a_wall_think 2>&1 | tee -a "$LOG"

echo "--- v8 thinking-OFF ---" | tee -a "$LOG"
python -u experiments/version_a_eval_panel.py \
  --checkpoints outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt \
  --n-harmful 64 --max-new-tokens 512 --gen-batch-size 64 --direction-layer 20 \
  --advbench-source walledai --qwen-thinking off --n-direction 256 --n-cap-eval 32 \
  --judge-workers 32 --run-id v8_wall_off 2>&1 | tee -a "$LOG"

echo "--- version_A thinking-OFF ---" | tee -a "$LOG"
python -u experiments/version_a_eval_panel.py \
  --checkpoints outputs/version_a_qwen_500.pt.s400.pt \
                outputs/version_a_qwen_500.pt.s450.pt \
                outputs/version_a_qwen_500.pt \
  --n-harmful 64 --max-new-tokens 512 --gen-batch-size 64 --direction-layer 20 \
  --advbench-source walledai --qwen-thinking off --n-direction 256 --n-cap-eval 32 \
  --judge-workers 32 --run-id version_a_wall_off 2>&1 | tee -a "$LOG"

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
