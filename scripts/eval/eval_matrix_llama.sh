#!/usr/bin/env bash
# Full eval matrix: Llama-3.2-1B base/v7/v8 x clean/attacked, across all benches.
# v7 = tamper_resistant_llama32_1b_v7_L13.pt ; v8 = tamper_resistant_llama32_1b_v8_best.pt (s425).
# DL=13. Materializes the 5 non-base dirs, then runs 9 evals/model (5 harm gens -> judge locally,
# ARC/MMLU-12/IFEval/GSM8K self-scored).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY=python; MID="meta-llama/Llama-3.2-1B-Instruct"; DL=13
V7=outputs/tamper_resistant_llama32_1b_v7_L13.pt
V8=outputs/tamper_resistant_llama32_1b_v8_best.pt
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
PDIR=scripts/external_benches/prompts
O=/workspace/outputs

echo "### materialize llama dirs ###"
$PY experiments/save_p1b_checkpoint.py --model-id "$MID"                 --attack all  --direction-layer $DL --out $O/llama_base_att
$PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$V7" --attack none --direction-layer $DL --out $O/llama_v7_clean
$PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$V7" --attack all  --direction-layer $DL --out $O/llama_v7_att
$PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$V8" --attack none --direction-layer $DL --out $O/llama_v8_clean
$PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$V8" --attack all  --direction-layer $DL --out $O/llama_v8_att

declare -A M=(
  [base_clean]="$MID" [base_att]="$O/llama_base_att"
  [v7_clean]="$O/llama_v7_clean" [v7_att]="$O/llama_v7_att"
  [v8_clean]="$O/llama_v8_clean" [v8_att]="$O/llama_v8_att"
)
saf(){ $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" --prompt-source "$3" \
  --advbench-source walledai --n-prompts "$4" --max-new-tokens 512 --max-length 4096 --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
saff(){ $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" --prompt-file "$3" \
  --n-prompts -1 --max-new-tokens 512 --max-length 4096 --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
cap(){ local p="$1" n="$2" a="pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9"
  lm_eval --model vllm --model_args "$a" --tasks arc_challenge --num_fewshot 0 --batch_size auto --output_path "results/lx_cap_${n}_arc"
  lm_eval --model vllm --model_args "$a" --tasks "$MMLU"       --num_fewshot 0 --batch_size auto --output_path "results/lx_cap_${n}_mmlu"
  lm_eval --model vllm --model_args "$a" --tasks ifeval --num_fewshot 0 --batch_size auto --apply_chat_template --output_path "results/lx_ifeval_${n}"
  lm_eval --model vllm --model_args "$a" --tasks gsm8k  --num_fewshot 5 --batch_size auto --output_path "results/lx_gsm8k_${n}"; }

for tag in "${!M[@]}"; do
  p="${M[$tag]}"; echo "############ $tag -> $p ############"
  saf  "$p" "lx_${tag}_advbench"  advbench  520
  saf  "$p" "lx_${tag}_harmbench" harmbench 200
  saff "$p" "lx_${tag}_strongreject"   "$PDIR/strongreject.jsonl"
  saff "$p" "lx_${tag}_jailbreakbench" "$PDIR/jailbreakbench.jsonl"
  saff "$p" "lx_${tag}_sorrybench"     "$PDIR/sorrybench.jsonl"
  cap  "$p" "$tag"
done
echo "### DONE — judge results/lx_*_{advbench,harmbench,strongreject,jailbreakbench,sorrybench} locally; cap in lx_cap_*/lx_ifeval_*/lx_gsm8k_* ###"
