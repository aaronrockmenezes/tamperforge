#!/usr/bin/env bash
# Judge version_C's heretic winners through the export path. KL cost is NOT defense --
# version_B's 5x KL story died exactly here, when t99 turned out to be 0.3212 harmful with
# GSM8K intact. These three are chosen to make the comparison direct:
#   t71  ref 3  kl 0.1448  -- same refusal count as version_B's t99, at 7.3x the KL
#   t156 ref 6  kl 0.1362  -- cheapest attack anywhere on version_C's front
#   t47  ref 2  kl 0.1500  -- lowest refusals, ~= version_B t17's KL (0.1397)
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/hvc_eval.log
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
echo "=== heretic-vs-version_C eval $(date -u) ===" | tee "$LOG"
echo "=== ref: version_B+t99 0.3212 harm / GSM8K 0.3700 | version_A+t175 0.2615 / 0.4064 ===" | tee -a "$LOG"

for T in 71 156 47; do
  D="outputs/heretic_vc_t${T}"
  echo "########## trial ${T} ##########" | tee -a "$LOG"
  python -u experiments/version_c_replay.py --checkpoint outputs/version_c_qwen_500.pt \
    --trial "t${T}" --params-json results/heretic_vc_trials.json \
    --direction-recipe heretic --application heretic_full --out "$D" 2>&1 | tee -a "$LOG"
  python -u experiments/p0_baseline_eval.py --run-id "hvc_t${T}" --model-id "$D" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking off 2>&1 | tee -a "$LOG"
  python -u experiments/judge_generations.py --generations "results/hvc_t${T}/generations.jsonl" \
    --run-id "hvc_t${T}_judged" --num-workers 32 2>&1 | tee -a "$LOG"
  a="pretrained=${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45"
  for spec in "arc:arc_challenge:0" "mmlu:${MMLU}:0" "gsm8k:gsm8k:5"; do
    nm="${spec%%:*}"; rest="${spec#*:}"; tk="${rest%:*}"; sh="${rest##*:}"
    o="results/hvccap_t${T}_${nm}"; [ -d "$o" ] && continue
    lm_eval --model vllm --model_args "$a" --tasks "$tk" --num_fewshot "$sh" \
      --batch_size auto --output_path "$o" 2>&1 | tee -a "$LOG" | grep -aE "\|acc|\|exact_match"
  done
done
echo "=== HVC DONE $(date -u) ===" | tee -a "$LOG"
