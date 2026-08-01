#!/usr/bin/env bash
# Heretic vs version_A s500: safety + capability across the Pareto front.
# Capability is decisive -- high harm with intact GSM8K = Heretic won; high harm with
# cratered GSM8K = MAD fired and the attacker only got a broken model.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/heretic_eval.log
echo "=== Heretic eval on version_A s500: $(date -u) ===" | tee "$LOG"

MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"

for T in t85 t175 t100; do
  D="outputs/heretic_va_s500_${T}"
  [ -d "$D" ] || { echo "!! missing $D" | tee -a "$LOG"; continue; }
  echo "########## $T -> $D ##########" | tee -a "$LOG"

  python -u experiments/p0_baseline_eval.py \
    --run-id "hva_${T}" --model-id "$D" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
    --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking off 2>&1 | tee -a "$LOG"

  python -u experiments/judge_generations.py \
    --generations "results/hva_${T}/generations.jsonl" \
    --run-id "hva_${T}_judged" --num-workers 32 2>&1 | tee -a "$LOG"

  a="pretrained=${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45"
  for spec in "arc:arc_challenge:0" "mmlu:${MMLU}:0" "gsm8k:gsm8k:5"; do
    name="${spec%%:*}"; rest="${spec#*:}"; tasks="${rest%:*}"; shots="${rest##*:}"
    out="results/hvacap_${T}_${name}"
    find "$out" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && continue
    lm_eval --model vllm --model_args "$a" --tasks "$tasks" --num_fewshot "$shots" \
      --batch_size auto --output_path "$out" 2>&1 | tee -a "$LOG" \
      | grep -aE "\|acc|\|exact_match" || echo "!! $name failed $T" | tee -a "$LOG"
  done
done
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
