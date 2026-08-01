#!/usr/bin/env bash
# SERIALIZED. Running lm_eval alongside heretic wedged a vLLM engine and leaked ~10.9GB of
# VRAM to a host-namespace PID we cannot kill. Nothing here runs concurrently.
# gpu_memory_utilization dropped 0.40 -> 0.30: only ~13.7GB of the card remains.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/final_chain.log
UTIL=0.30
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
echo "=== final chain $(date -u) ===" | tee "$LOG"

echo "### STAGE 1: IFEval (remaining arms) ###" | tee -a "$LOG"
for D in xtest_v8_surg_k16 xva_clean xva_rank1 xva_surg_k16 \
         xvb_clean xvb_rank1 xvb_surg_k16 heretic_va_s500_t85 heretic_va_s500_t175; do
  [ -d "outputs/$D" ] || continue
  o="results/ifev_${D}"; find "$o" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && continue
  echo "--- ifeval $D ---" | tee -a "$LOG"
  lm_eval --model vllm --model_args "pretrained=outputs/${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=${UTIL}" \
    --tasks ifeval --num_fewshot 0 --batch_size auto --output_path "$o" 2>&1 \
    | tee -a "$LOG" | grep -aE "inst_level_strict|prompt_level_strict"
done

echo "### STAGE 2: heretic-vs-version_B trials ###" | tee -a "$LOG"
for T in t17 t99 t65; do
  D="outputs/heretic_vb_${T}"
  [ -d "$D" ] || { echo "!! missing $D" | tee -a "$LOG"; continue; }
  echo "--- $T safety ---" | tee -a "$LOG"
  [ -f "results/hvb_${T}/generations.jsonl" ] || python -u experiments/p0_baseline_eval.py \
    --run-id "hvb_${T}" --model-id "$D" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
    --max-new-tokens 512 --max-length 4096 --backend vllm --vllm-batch-size 64 \
    --vllm-dtype bfloat16 --vllm-gpu-memory-utilization ${UTIL} \
    --vllm-temperature 0.0 --vllm-top-p 1.0 --qwen-thinking off 2>&1 | tee -a "$LOG"
  [ -f "results/hvb_${T}_judged/summary.json" ] || python -u experiments/judge_generations.py \
    --generations "results/hvb_${T}/generations.jsonl" --run-id "hvb_${T}_judged" \
    --num-workers 32 2>&1 | tee -a "$LOG"
  a="pretrained=${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=${UTIL}"
  for spec in "arc:arc_challenge:0" "mmlu:${MMLU}:0" "gsm8k:gsm8k:5" "ifeval:ifeval:0"; do
    nm="${spec%%:*}"; rest="${spec#*:}"; tk="${rest%:*}"; sh="${rest##*:}"
    o="results/hvbcap_${T}_${nm}"; find "$o" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && continue
    lm_eval --model vllm --model_args "$a" --tasks "$tk" --num_fewshot "$sh" \
      --batch_size auto --output_path "$o" 2>&1 | tee -a "$LOG" \
      | grep -aE "\|acc|\|exact_match|inst_level_strict"
  done
done
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"
