#!/usr/bin/env bash
# Full 9-bench matrix: gemma-3-1b base/v7/v8 x clean/attacked. v8 = gemma3_1b_v8_best.pt (s450).
# DL: base/v8 attack @14 (sweep peak), v7 @13 (v7's trained layer). Judge harm gens locally.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY=python; MID="google/gemma-3-1b-it"
V7=outputs/tamper_resistant_p1b_v7.pt
V8=outputs/tamper_resistant_gemma3_1b_v8_best.pt
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
PDIR=scripts/external_benches/prompts; O=/workspace/outputs
mat(){ [ -f "$2/model.safetensors" ] || $PY experiments/save_p1b_checkpoint.py --model-id "$MID" ${3:+--checkpoint "$3"} --attack "$4" --direction-layer "$5" --out "$2"; }
mat _ "$O/gm_base_att" ""   all  14
mat _ "$O/gm_v7_clean" "$V7" none 13
mat _ "$O/gm_v7_att"   "$V7" all  13
mat _ "$O/gm_v8_clean" "$V8" none 14
mat _ "$O/gm_v8_att"   "$V8" all  14
declare -A M=(
  [base_clean]="$MID" [base_att]="$O/gm_base_att"
  [v7_clean]="$O/gm_v7_clean" [v7_att]="$O/gm_v7_att"
  [v8_clean]="$O/gm_v8_clean" [v8_att]="$O/gm_v8_att"
)
saf(){ [ -f "results/$2/generations.jsonl" ] && { echo "skip $2"; return; }
  $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" --prompt-source "$3" \
  --advbench-source walledai --n-prompts "$4" --max-new-tokens 512 --max-length 4096 --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
saff(){ [ -f "results/$2/generations.jsonl" ] && { echo "skip $2"; return; }
  $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" --prompt-file "$3" \
  --n-prompts -1 --max-new-tokens 512 --max-length 4096 --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
cap(){ local p="$1" n="$2" a="pretrained=$1,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9"
  [ -f "results/gx_cap_${n}_arc/generations.jsonl" ] || lm_eval --model vllm --model_args "$a" --tasks arc_challenge --num_fewshot 0 --batch_size auto --output_path "results/gx_cap_${n}_arc"
  [ -f "results/gx_cap_${n}_mmlu/generations.jsonl" ] || lm_eval --model vllm --model_args "$a" --tasks "$MMLU" --num_fewshot 0 --batch_size auto --output_path "results/gx_cap_${n}_mmlu"
  [ -f "results/gx_ifeval_${n}/generations.jsonl" ] || lm_eval --model vllm --model_args "$a" --tasks ifeval --num_fewshot 0 --batch_size auto --apply_chat_template --output_path "results/gx_ifeval_${n}"
  [ -f "results/gx_gsm8k_${n}/generations.jsonl" ] || lm_eval --model vllm --model_args "$a" --tasks gsm8k --num_fewshot 5 --batch_size auto --output_path "results/gx_gsm8k_${n}"; }
for tag in "${!M[@]}"; do
  p="${M[$tag]}"; echo "########## gm_${tag} -> $p ##########"
  saf  "$p" "gx_${tag}_advbench"  advbench  520
  saf  "$p" "gx_${tag}_harmbench" harmbench 200
  saff "$p" "gx_${tag}_strongreject"   "$PDIR/strongreject.jsonl"
  saff "$p" "gx_${tag}_jailbreakbench" "$PDIR/jailbreakbench.jsonl"
  saff "$p" "gx_${tag}_sorrybench"     "$PDIR/sorrybench.jsonl"
  cap  "$p" "$tag"
done
echo "### 9-bench DONE — judge gx_*_{advbench,harmbench,strongreject,jailbreakbench,sorrybench} locally; cap in gx_cap_*/gx_ifeval_*/gx_gsm8k_* ###"
