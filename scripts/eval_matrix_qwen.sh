#!/usr/bin/env bash
# Full eval matrix: Qwen3-0.6B base/v7/v8 x clean/attacked, across all benches.
# Safety = gens (JUDGE LOCALLY after). Capability = lm_eval (self-scoring).
# Materialize the 4 non-base dirs first (base = HF id):
#   v7 DL=14, v8 DL=20 (each attacked at its own trained layer).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY=python
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
PDIR=scripts/external_benches/prompts

# tag -> model path
declare -A M=(
  [base_clean]="Qwen/Qwen3-0.6B"
  [base_att]="/workspace/outputs/qwen_base_att"
  [v7_clean]="/workspace/outputs/qwen_v7_clean"
  [v7_att]="/workspace/outputs/qwen_v7_att"
  [v8_clean]="/workspace/outputs/qwen_v8_clean"
  [v8_att]="/workspace/outputs/qwen_v8_att"
)

saf(){ $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
  --prompt-source "$3" --advbench-source walledai --n-prompts "$4" --max-new-tokens 512 \
  --max-length 4096 --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
saff(){ $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
  --prompt-file "$3" --n-prompts -1 --max-new-tokens 512 --max-length 4096 \
  --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
cap(){ local p="$1" n="$2" a="pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9"
  lm_eval --model vllm --model_args "$a" --tasks arc_challenge --num_fewshot 0 --batch_size auto --output_path "results/mx_cap_${n}_arc"
  lm_eval --model vllm --model_args "$a" --tasks "$MMLU"       --num_fewshot 0 --batch_size auto --output_path "results/mx_cap_${n}_mmlu"
  lm_eval --model vllm --model_args "$a" --tasks ifeval --num_fewshot 0 --batch_size auto --apply_chat_template --output_path "results/mx_ifeval_${n}"
  lm_eval --model vllm --model_args "$a" --tasks gsm8k  --num_fewshot 5 --batch_size auto --output_path "results/mx_gsm8k_${n}"; }

for tag in "${!M[@]}"; do
  p="${M[$tag]}"
  echo "############ $tag -> $p ############"
  saf  "$p" "mx_${tag}_advbench"    advbench    520
  saf  "$p" "mx_${tag}_harmbench"   harmbench   200
  saff "$p" "mx_${tag}_strongreject"   "$PDIR/strongreject.jsonl"
  saff "$p" "mx_${tag}_jailbreakbench" "$PDIR/jailbreakbench.jsonl"
  saff "$p" "mx_${tag}_sorrybench"     "$PDIR/sorrybench.jsonl"
  cap  "$p" "$tag"
done
echo "### DONE — judge results/mx_*_{advbench,harmbench,beavertails,strongreject,jailbreakbench,sorrybench} locally; cap in results/mx_cap_*, mx_ifeval_* ###"
