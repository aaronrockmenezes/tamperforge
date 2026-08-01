#!/usr/bin/env bash
# Direction-layer sweep at fixed alpha=1.0. Strength is ruled out (flat 0.0000 harmful
# across 0.2-1.5), so the suspects are WHERE the direction is read from and WHICH matrices
# are hit. Heretic's winning trials all read from 12.6-14.2; training only used {16,20,24}.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/dl_sweep.log
CK=outputs/version_a_qwen_500.pt
echo "=== direction-layer sweep, alpha=1.0: $(date -u) ===" | tee "$LOG"
echo "=== control: DL=20 must reproduce rank-1 (0.0000 harmful, GSM8K ~0.036) ===" | tee -a "$LOG"

run_one () {  # $1=tag $2=extra args
  local TAG=$1; shift
  local D="outputs/dlsw_${TAG}"
  [ -f "$D/model.safetensors" ] || python -u experiments/version_a_alpha_sweep.py \
      --checkpoint "$CK" --alpha 1.0 --out "$D" "$@" 2>&1 | tee -a "$LOG"
  [ -f "results/dlsw_${TAG}/generations.jsonl" ] || python -u experiments/p0_baseline_eval.py \
    --run-id "dlsw_${TAG}" --model-id "$D" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
    --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking off 2>&1 | tee -a "$LOG"
  [ -f "results/dlsw_${TAG}_judged/summary.json" ] || python -u experiments/judge_generations.py \
    --generations "results/dlsw_${TAG}/generations.jsonl" \
    --run-id "dlsw_${TAG}_judged" --num-workers 32 2>&1 | tee -a "$LOG"
  local out="results/dlswcap_${TAG}_gsm8k"
  find "$out" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . || lm_eval --model vllm \
    --model_args "pretrained=${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45" \
    --tasks gsm8k --num_fewshot 5 --batch_size auto --output_path "$out" 2>&1 \
    | tee -a "$LOG" | grep -aE "\|gsm8k"
  rm -rf "$D"
}

for L in 8 12 14 16 20 24 26; do
  echo "########## direction_layer=$L (read+write) ##########" | tee -a "$LOG"
  run_one "L${L}" --direction-layer "$L"
done

# write-only at the layers that matter most: Heretic's winning band vs our trained one
for L in 14 20; do
  echo "########## direction_layer=$L WRITE-ONLY ##########" | tee -a "$LOG"
  run_one "L${L}wo" --direction-layer "$L" --scope write_only
done

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"
