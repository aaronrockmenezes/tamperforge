#!/usr/bin/env bash
# Collapse-trigger curve: vary ONLY ablation strength on the canonical rank-1 shape.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/alpha_sweep.log
CK=outputs/version_a_qwen_500.pt
echo "=== alpha sweep on version_A s500: $(date -u) ===" | tee "$LOG"
echo "=== control: alpha=1.0 must reproduce rank-1 (0.0000 harmful, GSM8K 0.0394) ===" | tee -a "$LOG"

for A in 0.2 0.4 0.6 0.8 1.0 1.2 1.5; do
  TAG="a$(echo $A | tr -d '.')"
  D="outputs/asweep_${TAG}"
  echo "########## alpha=$A ##########" | tee -a "$LOG"
  [ -f "$D/model.safetensors" ] || python -u experiments/version_a_alpha_sweep.py \
      --checkpoint "$CK" --alpha "$A" --out "$D" 2>&1 | tee -a "$LOG"

  [ -f "results/asw_${TAG}/generations.jsonl" ] || python -u experiments/p0_baseline_eval.py \
    --run-id "asw_${TAG}" --model-id "$D" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
    --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking off 2>&1 | tee -a "$LOG"

  [ -f "results/asw_${TAG}_judged/summary.json" ] || python -u experiments/judge_generations.py \
    --generations "results/asw_${TAG}/generations.jsonl" \
    --run-id "asw_${TAG}_judged" --num-workers 32 2>&1 | tee -a "$LOG"

  out="results/aswcap_${TAG}_gsm8k"
  find "$out" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . || lm_eval --model vllm \
    --model_args "pretrained=${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45" \
    --tasks gsm8k --num_fewshot 5 --batch_size auto --output_path "$out" 2>&1 \
    | tee -a "$LOG" | grep -aE "\|gsm8k"

  rm -rf "$D"   # 1.2GB each, 7 of them; meta is kept in results
done
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"
