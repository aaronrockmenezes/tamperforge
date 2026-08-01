#!/usr/bin/env bash
# Close the clean-quality gap: v8 and version_A clean, BOTH thinking modes, validated path.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/clean_matrix.log
echo "=== clean matrix (export path, 520 prompts): $(date -u) ===" | tee "$LOG"

run_eval () {  # $1=tag  $2=model dir  $3=thinking
  python -u experiments/p0_baseline_eval.py \
    --run-id "xcm_$1" --model-id "$2" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
    --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking "$3" 2>&1 | tee -a "$LOG"
  python -u experiments/judge_generations.py \
    --generations "results/xcm_$1/generations.jsonl" \
    --run-id "xcm_$1_judged" --num-workers 32 2>&1 | tee -a "$LOG"
}

# v8 clean materialised once, evaluated in both modes
echo "--- exporting v8 clean ---" | tee -a "$LOG"
python -u experiments/save_p1b_checkpoint.py \
  --checkpoint outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt \
  --model-id Qwen/Qwen3-0.6B --attack none --out outputs/xcm_v8_clean 2>&1 | tee -a "$LOG"

echo "--- [1/2] v8 clean, thinking-ON (matches the surgical evals) ---" | tee -a "$LOG"
run_eval v8_clean_think outputs/xcm_v8_clean default

echo "--- [2/2] version_A s500 clean, thinking-OFF ---" | tee -a "$LOG"
run_eval va_clean_off outputs/xva_clean off

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
