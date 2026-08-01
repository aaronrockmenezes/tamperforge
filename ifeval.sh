#!/usr/bin/env bash
# IFEval across every arm. This is the only GENERATIVE general-capability measure in the
# suite: ARC/MMLU are loglikelihood, so a model that generates garbage still scores fine on
# them, and GSM8K is math-only. The campaign recipe includes IFEval for exactly this reason
# -- it overturned Llama's "zero capability cost" read once already.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/ifeval_all.log
echo "=== IFEval sweep: $(date -u) ===" | tee "$LOG"
for D in xcm_v8_clean xtest_v8_surg_k16 \
         xva_clean xva_rank1 xva_surg_k16 \
         xvb_clean xvb_rank1 xvb_surg_k16 \
         heretic_va_s500_t85 heretic_va_s500_t175; do
  [ -d "outputs/$D" ] || { echo "!! missing $D" | tee -a "$LOG"; continue; }
  o="results/ifev_${D}"
  find "$o" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && continue
  echo "########## $D ##########" | tee -a "$LOG"
  lm_eval --model vllm \
    --model_args "pretrained=outputs/${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.40" \
    --tasks ifeval --num_fewshot 0 --batch_size auto --output_path "$o" 2>&1 \
    | tee -a "$LOG" | grep -aE "inst_level_strict|prompt_level_strict"
done
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
