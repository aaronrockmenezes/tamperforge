#!/usr/bin/env bash
# Four-cell Qwen3-8B thinking-mode eval:
#   base_clean, base_att, v8_clean, v8_att
#
# Required:
#   DL=<best layer> BEST_CKPT=<picked .pt> bash scripts/qwen3_8b_thinking_eval_4cell.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DL="${DL:?set DL}"
BEST_CKPT="${BEST_CKPT:?set BEST_CKPT to picked snapshot .pt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
BATCH="${BATCH:-32}"
GPU_MEM="${GPU_MEM:-0.82}"
MMLU="${MMLU:-mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security}"

export TF_QWEN_THINKING=on
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PY" experiments/save_p1b_checkpoint.py \
  --model-id "$MODEL" \
  --attack all \
  --direction-layer "$DL" \
  --qwen-thinking on \
  --out outputs/qwen3_8b_thinking_base_att

"$PY" experiments/save_p1b_checkpoint.py \
  --model-id "$MODEL" \
  --checkpoint "$BEST_CKPT" \
  --attack none \
  --direction-layer "$DL" \
  --qwen-thinking on \
  --out outputs/qwen3_8b_thinking_v8_clean

"$PY" experiments/save_p1b_checkpoint.py \
  --model-id "$MODEL" \
  --checkpoint "$BEST_CKPT" \
  --attack all \
  --direction-layer "$DL" \
  --qwen-thinking on \
  --out outputs/qwen3_8b_thinking_v8_att

declare -A MODELS=(
  [base_clean]="$MODEL"
  [base_att]="outputs/qwen3_8b_thinking_base_att"
  [v8_clean]="outputs/qwen3_8b_thinking_v8_clean"
  [v8_att]="outputs/qwen3_8b_thinking_v8_att"
)

safety() {
  local model_path="$1"
  local run_id="$2"
  "$PY" experiments/p0_baseline_eval.py \
    --backend vllm \
    --model-id "$model_path" \
    --prompt-source advbench \
    --advbench-source walledai \
    --n-prompts 520 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-length "$MAX_LENGTH" \
    --vllm-batch-size "$BATCH" \
    --vllm-gpu-memory-utilization "$GPU_MEM" \
    --qwen-thinking on \
    --vllm-temperature 0.6 \
    --vllm-top-p 0.95 \
    --vllm-top-k 20 \
    --n-arc 0 \
    --run-id "$run_id"
  "$PY" experiments/judge_generations.py \
    --generations "results/${run_id}/generations.jsonl" \
    --run-id "${run_id}_judged" \
    --num-workers "${JUDGE_WORKERS:-32}"
}

capability() {
  local model_path="$1"
  local tag="$2"
  local args="pretrained=${model_path},dtype=bfloat16,trust_remote_code=True,max_model_len=${MAX_LENGTH},gpu_memory_utilization=${GPU_MEM}"
  lm_eval --model vllm --model_args "$args" --tasks arc_challenge --num_fewshot 0 --batch_size auto --output_path "results/q8t_cap_${tag}_arc"
  lm_eval --model vllm --model_args "$args" --tasks "$MMLU" --num_fewshot 0 --batch_size auto --output_path "results/q8t_cap_${tag}_mmlu"
  lm_eval --model vllm --model_args "$args" --tasks ifeval --num_fewshot 0 --batch_size auto --apply_chat_template --output_path "results/q8t_ifeval_${tag}"
  lm_eval --model vllm --model_args "$args" --tasks gsm8k --num_fewshot 5 --batch_size auto --output_path "results/q8t_gsm8k_${tag}"
}

for tag in base_clean base_att v8_clean v8_att; do
  echo "############ ${tag} -> ${MODELS[$tag]} ############"
  safety "${MODELS[$tag]}" "q8t_${tag}_advbench520"
  capability "${MODELS[$tag]}" "$tag"
done

echo "### done. Safety summaries are results/q8t_*_advbench520_judged/summary.json; capability is results/q8t_*."
