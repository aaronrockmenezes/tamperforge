#!/usr/bin/env bash
# Extended-bench matrix: Qwen3-0.6B base/v7/v8 x clean/attacked, NEW benches only.
# XSTest(safe/unsafe) + OR-Bench-Hard + SimpleQA = prompt-file gens (score locally:
# xstest/orbench via judge_generations refusal-rate, simpleqa via score_simpleqa.py).
# MBPP = lm_eval self-score (code execution). Materializes the 5 non-base dirs if missing.
# DL: base/v8 = 20 (qwen peak), v7 = 14 (its product layer).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_ALLOW_CODE_EVAL=1   # MBPP code_eval metric requires this (sandboxed vast box)
PY=python; MID="Qwen/Qwen3-0.6B"
V7=outputs/tamper_resistant_qwen3_0p6b_v7.pt
V8=outputs/tamper_resistant_qwen3_0p6b_v8.pt
PDIR=scripts/external_benches/prompts; O=/workspace/outputs
mat(){ [ -d "$2" ] || $PY experiments/save_p1b_checkpoint.py --model-id "$MID" ${3:+--checkpoint "$3"} --attack "$4" --direction-layer "$5" --out "$2"; }
mat _ "$O/nq_base_att"  ""   all  20
mat _ "$O/nq_v7_clean"  "$V7" none 14
mat _ "$O/nq_v7_att"    "$V7" all  14
mat _ "$O/nq_v8_clean"  "$V8" none 20
mat _ "$O/nq_v8_att"    "$V8" all  20

declare -A M=(
  [base_clean]="$MID" [base_att]="$O/nq_base_att"
  [v7_clean]="$O/nq_v7_clean" [v7_att]="$O/nq_v7_att"
  [v8_clean]="$O/nq_v8_clean" [v8_att]="$O/nq_v8_att"
)
gen(){ $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" --prompt-file "$3" \
  --n-prompts -1 --max-new-tokens 512 --max-length 4096 --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }

for tag in "${!M[@]}"; do
  p="${M[$tag]}"; echo "############ $tag -> $p ############"
  gen "$p" "nq_${tag}_xstest_safe"   "$PDIR/xstest_safe.jsonl"
  gen "$p" "nq_${tag}_xstest_unsafe" "$PDIR/xstest_unsafe.jsonl"
  gen "$p" "nq_${tag}_orbench"       "$PDIR/orbench.jsonl"
  gen "$p" "nq_${tag}_simpleqa"      "$PDIR/simpleqa.jsonl"
  lm_eval --model vllm --model_args "pretrained=${p},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9" \
    --tasks mbpp --num_fewshot 3 --batch_size auto --confirm_run_unsafe_code --output_path "results/nq_mbpp_${tag}" \
    || echo "!! mbpp failed for $tag (non-fatal, continuing)"
done
echo "### DONE. Score locally:"
echo "  xstest/orbench -> judge_generations (refusal-rate: SAFE splits want LOW = not over-refusing)"
echo "  simpleqa       -> scripts/external_benches/score_simpleqa.py"
echo "  mbpp           -> results/nq_mbpp_* (pass@1, self-scored) ###"
