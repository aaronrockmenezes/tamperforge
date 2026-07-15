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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
MAX_LENGTH="${MAX_LENGTH:-20480}"
BATCH="${BATCH:-24}"
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
  local base_args="pretrained=${model_path},dtype=bfloat16,trust_remote_code=True,max_model_len=${MAX_LENGTH},gpu_memory_utilization=${GPU_MEM}"
  local score_args="${base_args},enable_thinking=False"
  local think_args="${base_args},enable_thinking=True,think_end_token=</think>,max_gen_toks=${MAX_NEW_TOKENS}"

  # ARC/MMLU are log-likelihood tasks. lm-eval explicitly disallows reasoning
  # mode for them, so use Qwen's non-thinking chat template and label them as
  # non-thinking capability axes. Generative IFEval/GSM8K below are the actual
  # thinking-mode capability tests; lm-eval strips the trace at </think> before
  # scoring the final answer.
  lm_eval run --model vllm --model_args "$score_args" --tasks arc_challenge \
    --num_fewshot 0 --batch_size auto --max_batch_size "$BATCH" --apply_chat_template \
    --output_path "results/q8t_cap_nonthink_${tag}_arc"
  lm_eval run --model vllm --model_args "$score_args" --tasks "$MMLU" \
    --num_fewshot 0 --batch_size auto --max_batch_size "$BATCH" --apply_chat_template \
    --output_path "results/q8t_cap_nonthink_${tag}_mmlu"
  lm_eval run --model vllm --model_args "$think_args" --tasks ifeval \
    --num_fewshot 0 --batch_size auto --max_batch_size "$BATCH" --apply_chat_template \
    --gen_kwargs "max_gen_toks=${MAX_NEW_TOKENS}" "do_sample=True" \
      "temperature=0.6" "top_p=0.95" "top_k=20" \
    --output_path "results/q8t_thinking_${tag}_ifeval" --log_samples
  lm_eval run --model vllm --model_args "$think_args" --tasks gsm8k \
    --num_fewshot 5 --batch_size auto --max_batch_size "$BATCH" --apply_chat_template \
    --gen_kwargs "max_gen_toks=${MAX_NEW_TOKENS}" "do_sample=True" \
      "temperature=0.6" "top_p=0.95" "top_k=20" \
    --output_path "results/q8t_thinking_${tag}_gsm8k" --log_samples
}

for tag in base_clean base_att v8_clean v8_att; do
  echo "############ ${tag} -> ${MODELS[$tag]} ############"
  safety "${MODELS[$tag]}" "q8t_${tag}_advbench520"
  capability "${MODELS[$tag]}" "$tag"
done

echo "### done. Safety summaries are results/q8t_*_advbench520_judged/summary.json; capability is results/q8t_*."
