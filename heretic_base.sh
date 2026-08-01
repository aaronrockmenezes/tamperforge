#!/usr/bin/env bash
# Heretic vs BASE Qwen3-0.6B -- the missing ceiling control.
#
# Without this we cannot interpret heretic's ~0.32 harmful against version_B/C. The
# uncensoring range on this model is base-clean 0.2577 -> base-rank1 0.6385, so 0.32 is
# either "heretic constrained to ~17% of headroom" (defense partly working) or "heretic
# does about this everywhere" (defense irrelevant). Only heretic-vs-base separates them.
#
# 200 trials, same as the version_B and version_C runs, so the fronts are comparable.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/heretic_base_eval.log
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
echo "=== heretic vs BASE Qwen3-0.6B $(date -u) ===" | tee "$LOG"
echo "=== range: base clean 0.2577 -> base rank1 0.6385 | heretic vs vB 0.3212, vC 0.3231 ===" | tee -a "$LOG"

# base must live on disk for heretic to attack it
if [ ! -d outputs/xbase_clean_hf ]; then
  python - <<'PY' 2>&1 | tee -a "$LOG"
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype="bfloat16")
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
m.save_pretrained("outputs/xbase_clean_hf", safe_serialization=True); t.save_pretrained("outputs/xbase_clean_hf")
print("[base] materialised to outputs/xbase_clean_hf")
PY
fi

rm -rf /tmp/hcp_base
heretic --model outputs/xbase_clean_hf --n-trials 200 --seed 0 \
  --study-checkpoint-dir /tmp/hcp_base < /dev/null \
  > logs/training_runs/heretic_base_trials.log 2>&1
echo "heretic rc=$? (rc=1 expected: EOF at its menu)" | tee -a "$LOG"
echo "trials: $(grep -ac 'Running trial' logs/training_runs/heretic_base_trials.log)" | tee -a "$LOG"
echo "=== HERETIC-BASE STUDY DONE $(date -u) ===" | tee -a "$LOG"
