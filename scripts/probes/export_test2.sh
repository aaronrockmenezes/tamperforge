#!/usr/bin/env bash
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/export_test_v8_surg_k16.log
echo "=== EVAL RETRY (with --n-arc 0): $(date -u) ===" | tee -a "$LOG"

python -u experiments/p0_baseline_eval.py \
  --run-id xtest_v8_surg_k16 \
  --model-id outputs/xtest_v8_surg_k16 \
  --prompt-source advbench --advbench-source walledai --advbench-split train \
  --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
  --max-new-tokens 512 --max-length 4096 \
  --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
  --vllm-gpu-memory-utilization 0.45 \
  --vllm-temperature 0.0 --vllm-top-p 1.0 \
  --qwen-thinking default 2>&1 | tee -a "$LOG"

python -u experiments/judge_generations.py \
  --generations results/xtest_v8_surg_k16/generations.jsonl \
  --run-id xtest_v8_surg_k16_judged --num-workers 32 2>&1 | tee -a "$LOG"

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
