#!/usr/bin/env bash
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/version_a_panel_full.log
echo "=== version_A full panel: $(date -u) ===" | tee "$LOG"
python -u experiments/version_a_eval_panel.py \
  --checkpoints outputs/version_a_qwen_500.pt.s400.pt \
                outputs/version_a_qwen_500.pt.s450.pt \
                outputs/version_a_qwen_500.pt \
  --n-harmful 64 --max-new-tokens 512 --gen-batch-size 64 \
  --n-direction 256 --n-cap-eval 32 \
  --judge-workers 8 --run-id version_a_panel_full 2>&1 | tee -a "$LOG"
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"
