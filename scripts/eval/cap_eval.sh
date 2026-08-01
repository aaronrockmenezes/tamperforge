#!/usr/bin/env bash
# Capability under attack -- the MAD claim proper. Matches scripts/v11_cap_eval.sh exactly
# (ARC 0-shot, same MMLU 12-subject subset, GSM8K 5-shot) so numbers compare to the
# campaign's v8 figures. gpu_memory_utilization lowered from 0.85: a foreign process holds
# ~8.7GB on this card.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/cap_eval_version_a.log
echo "=== version_A capability eval: $(date -u) ===" | tee "$LOG"

MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"

declare -A M=(
  [va_clean]="outputs/xva_clean"
  [va_rank1]="outputs/xva_rank1"
  [va_surg_k16]="outputs/xva_surg_k16"
)

for tag in va_clean va_rank1 va_surg_k16; do
  p="${M[$tag]}"
  [ -d "$p" ] || { echo "!! missing $p" | tee -a "$LOG"; continue; }
  echo "########## $tag -> $p ##########" | tee -a "$LOG"
  a="pretrained=${p},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45"
  for spec in "arc:arc_challenge:0" "mmlu:${MMLU}:0" "gsm8k:gsm8k:5"; do
    name="${spec%%:*}"; rest="${spec#*:}"; tasks="${rest%:*}"; shots="${rest##*:}"
    out="results/vacap_${tag}_${name}"
    find "$out" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && continue
    lm_eval --model vllm --model_args "$a" --tasks "$tasks" \
      --num_fewshot "$shots" --batch_size auto --output_path "$out" 2>&1 \
      | tee -a "$LOG" | grep -aE "\|acc|\|exact_match" || echo "!! $name failed $tag" | tee -a "$LOG"
  done
done
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
