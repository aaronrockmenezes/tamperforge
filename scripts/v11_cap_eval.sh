#!/usr/bin/env bash
# Capability eval for the v11 surgical-ablation arms, matching the campaign config exactly
# (scripts/eval_matrix_gemma.sh): ARC 0-shot, the same MMLU 12-subject subset, GSM8K 5-shot.
#
# This decides how bad gate 2 is. Surgical ablation already gets 44.8% coherent harm on v8
# where plain rank-1 gets 0.0%. If it ALSO keeps capability, MAD is fully broken -- the
# attacker gets smart AND dangerous. If it pays a capability cost, a weaker residual claim
# survives ("the attacker must sacrifice X%").
#
#   PY=/venv/main/bin/python bash scripts/v11_cap_eval.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"

# v8_clean is the reference the attacked arms must be compared against, not base.
declare -A M=(
  [v8_clean]="outputs/v11_qwen_v8_clean"
  [v8_surg_k0]="outputs/v11_v8_surg_k0"
  [v8_surg_k16]="outputs/v11_v8_surg_k16"
  [v8_surg_k64]="outputs/v11_v8_surg_k64"
)

for tag in v8_clean v8_surg_k0 v8_surg_k16 v8_surg_k64; do
  p="${M[$tag]}"
  [ -d "$p" ] || { echo "!! missing $p, skipping $tag"; continue; }
  echo "############ $tag -> $p ############"
  a="pretrained=${p},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.85"
  [ -f "results/v11cap_${tag}_arc/generations.jsonl" ]  || lm_eval --model vllm --model_args "$a" \
      --tasks arc_challenge --num_fewshot 0 --batch_size auto \
      --output_path "results/v11cap_${tag}_arc"  || echo "!! arc failed $tag"
  [ -f "results/v11cap_${tag}_mmlu/generations.jsonl" ] || lm_eval --model vllm --model_args "$a" \
      --tasks "$MMLU" --num_fewshot 0 --batch_size auto \
      --output_path "results/v11cap_${tag}_mmlu" || echo "!! mmlu failed $tag"
  [ -f "results/v11cap_${tag}_gsm8k/generations.jsonl" ] || lm_eval --model vllm --model_args "$a" \
      --tasks gsm8k --num_fewshot 5 --batch_size auto \
      --output_path "results/v11cap_${tag}_gsm8k" || echo "!! gsm8k failed $tag"
done
echo "### v11 capability eval DONE ###"
