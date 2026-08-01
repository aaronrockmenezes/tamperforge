#!/usr/bin/env bash
# Uses the repo's existing vLLM-shutdown-hang workaround (scripts/run_mad_v10_s175_vllm_caps.sh,
# commit 1a4e603): some vLLM versions hang AFTER lm-eval has written its final result, so the
# process never exits and a sequential loop stalls forever. Run in background, poll for the
# result file, then stop that evaluator's EngineCore.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/final_chain2.log
UTIL=0.75
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
echo "=== chain2 $(date -u) ===" | tee "$LOG"

result_exists() { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }
cleanup_leftovers() { pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true; sleep 5; }

run_lm () {   # $1=outdir  $2=model dir  $3=tasks  $4=fewshot
  local out="$1" md="$2" tasks="$3" shots="$4"
  result_exists "$out" && { echo "[skip] $out" | tee -a "$LOG"; return 0; }
  echo "--- $out ---" | tee -a "$LOG"
  lm_eval --model vllm \
    --model_args "pretrained=${md},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=${UTIL}" \
    --tasks "$tasks" --num_fewshot "$shots" --batch_size auto --output_path "$out" >>"$LOG" 2>&1 &
  local pid=$!
  local waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if result_exists "$out"; then
      sleep 5
      local ep; ep="$(pgrep -P "$pid" -f "VLLM::EngineCore" | head -1 || true)"
      [ -n "$ep" ] && { echo "[cleanup] stopping finished EngineCore $ep" | tee -a "$LOG"; kill -TERM "$ep" 2>/dev/null || true; }
      sleep 10; kill -TERM "$pid" 2>/dev/null || true
      break
    fi
    sleep 10; waited=$((waited+10))
    [ "$waited" -gt 1800 ] && { echo "[timeout] $out" | tee -a "$LOG"; kill -TERM "$pid" 2>/dev/null; break; }
  done
  wait "$pid" 2>/dev/null || true
  cleanup_leftovers
  grep -aE "inst_level_strict|prompt_level_strict|\|acc |\|exact_match" "$LOG" | tail -3
}

echo "### STAGE 1: IFEval ###" | tee -a "$LOG"
for D in xva_clean xva_rank1 xva_surg_k16 xvb_clean xvb_rank1 xvb_surg_k16 \
         heretic_va_s500_t85 heretic_va_s500_t175; do
  [ -d "outputs/$D" ] && run_lm "results/ifev_${D}" "outputs/${D}" ifeval 0
done

echo "### STAGE 2: heretic vs version_B ###" | tee -a "$LOG"
for T in t17 t99 t65; do
  D="outputs/heretic_vb_${T}"; [ -d "$D" ] || continue
  if [ ! -d "results/hvb_${T}" ]; then
    echo "--- $T safety ---" | tee -a "$LOG"
    python -u experiments/p0_baseline_eval.py --run-id "hvb_${T}" --model-id "$D" \
      --prompt-source advbench --advbench-source walledai --advbench-split train \
      --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
      --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
      --vllm-gpu-memory-utilization ${UTIL} --vllm-temperature 0.0 --vllm-top-p 1.0 \
      --qwen-thinking off >>"$LOG" 2>&1
    cleanup_leftovers
  fi
  [ -f "results/hvb_${T}_judged/summary.json" ] || python -u experiments/judge_generations.py \
    --generations "results/hvb_${T}/generations.jsonl" --run-id "hvb_${T}_judged" \
    --num-workers 32 >>"$LOG" 2>&1
  run_lm "results/hvbcap_${T}_arc"    "$D" arc_challenge 0
  run_lm "results/hvbcap_${T}_mmlu"   "$D" "$MMLU" 0
  run_lm "results/hvbcap_${T}_gsm8k"  "$D" gsm8k 5
  run_lm "results/hvbcap_${T}_ifeval" "$D" ifeval 0
done
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
