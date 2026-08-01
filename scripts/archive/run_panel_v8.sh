#!/usr/bin/env bash
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/v8_panel_control.log
echo "=== v8 CONTROL panel (identical config to version_A): $(date -u) ===" | tee "$LOG"
python -u experiments/version_a_eval_panel.py \
  --checkpoints outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt \
  --n-harmful 64 --max-new-tokens 512 --gen-batch-size 64 \
  --n-direction 256 --n-cap-eval 32 \
  --judge-workers 32 --run-id v8_panel_control 2>&1 | tee -a "$LOG"
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
