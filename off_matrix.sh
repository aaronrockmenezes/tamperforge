#!/usr/bin/env bash
# The cell that decides it: version_A ATTACKED at thinking-OFF, the trained regime.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/off_matrix.log
echo "=== attacked @ thinking-OFF: $(date -u) ===" | tee "$LOG"

run_eval () {
  python -u experiments/p0_baseline_eval.py \
    --run-id "xoff_$1" --model-id "$2" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
    --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking off 2>&1 | tee -a "$LOG"
  python -u experiments/judge_generations.py \
    --generations "results/xoff_$1/generations.jsonl" \
    --run-id "xoff_$1_judged" --num-workers 32 2>&1 | tee -a "$LOG"
}

echo "--- [1/3] version_A surgical k16, thinking-OFF ---" | tee -a "$LOG"
run_eval va_surg_k16 outputs/xva_surg_k16
echo "--- [2/3] version_A rank-1, thinking-OFF ---" | tee -a "$LOG"
run_eval va_rank1 outputs/xva_rank1
echo "--- [3/3] v8 surgical k16, thinking-OFF (control) ---" | tee -a "$LOG"
run_eval v8_surg_k16 outputs/xtest_v8_surg_k16

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
