#!/usr/bin/env bash
# Parameterized extended-bench matrix (XSTest/OR-Bench/SimpleQA/MBPP) for ANY model family,
# base/v7/v8 x clean/attacked. Generalization of eval_matrix_new_qwen.sh.
# Score locally after: xstest/orbench -> score_overrefusal.py ; simpleqa -> score_simpleqa.py ; mbpp self-scored.
#   MID=meta-llama/Llama-3.2-1B-Instruct TAG=ll V7=outputs/tamper_resistant_llama32_1b_v7_L13.pt \
#     V8=outputs/tamper_resistant_llama32_1b_v8_best.pt DLBASE=13 DLV7=13 DLV8=13 \
#     CUDA_VISIBLE_DEVICES=0 bash scripts/eval_matrix_new.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ALLOW_CODE_EVAL=1
PY=python
MID="${MID:?}"; TAG="${TAG:?short prefix e.g. ll/gm}"; V7="${V7:?}"; V8="${V8:?}"
DLBASE="${DLBASE:?}"; DLV7="${DLV7:?}"; DLV8="${DLV8:?}"
PDIR=scripts/external_benches/prompts; O=/workspace/outputs
mat(){ [ -f "$2/model.safetensors" ] || $PY experiments/save_p1b_checkpoint.py --model-id "$MID" ${3:+--checkpoint "$3"} --attack "$4" --direction-layer "$5" --out "$2"; }
mat _ "$O/${TAG}_base_att" ""   all  "$DLBASE"
mat _ "$O/${TAG}_v7_clean" "$V7" none "$DLV7"
mat _ "$O/${TAG}_v7_att"   "$V7" all  "$DLV7"
mat _ "$O/${TAG}_v8_clean" "$V8" none "$DLV8"
mat _ "$O/${TAG}_v8_att"   "$V8" all  "$DLV8"
declare -A M=(
  [base_clean]="$MID" [base_att]="$O/${TAG}_base_att"
  [v7_clean]="$O/${TAG}_v7_clean" [v7_att]="$O/${TAG}_v7_att"
  [v8_clean]="$O/${TAG}_v8_clean" [v8_att]="$O/${TAG}_v8_att"
)
gen(){ [ -f "results/$2/generations.jsonl" ] && { echo "skip $2"; return; }
  $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" --prompt-file "$3" \
  --n-prompts -1 --max-new-tokens 512 --max-length 4096 --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
for tag in "${!M[@]}"; do
  p="${M[$tag]}"; echo "########## ${TAG}_${tag} -> $p ##########"
  gen "$p" "nq${TAG}_${tag}_xstest_safe"   "$PDIR/xstest_safe.jsonl"
  gen "$p" "nq${TAG}_${tag}_xstest_unsafe" "$PDIR/xstest_unsafe.jsonl"
  gen "$p" "nq${TAG}_${tag}_orbench"       "$PDIR/orbench.jsonl"
  gen "$p" "nq${TAG}_${tag}_simpleqa"      "$PDIR/simpleqa.jsonl"
  if [ -f "results/nq${TAG}_mbpp_${tag}/generations.jsonl" ]; then echo "skip mbpp $tag"; else
    lm_eval --model vllm --model_args "pretrained=${p},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9" \
      --tasks mbpp --num_fewshot 3 --batch_size auto --confirm_run_unsafe_code --output_path "results/nq${TAG}_mbpp_${tag}" \
      || echo "!! mbpp failed $tag"; fi
done
echo "### DONE. Score locally: score_overrefusal.py (xstest/orbench), score_simpleqa.py (simpleqa); mbpp self-scored. ###"
