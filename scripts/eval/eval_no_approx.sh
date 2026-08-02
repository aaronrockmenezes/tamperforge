#!/usr/bin/env bash
# Full eval of the REAL heretic checkpoints (no replay approximation) saved from
# the seed-1 study against version_B-Llama: trials 24 and 63.
#
# Also fills the two code-benchmark reference cells that don't exist yet
# (llama base + lvb_clean on humaneval/mbpp) so the heretic numbers mean something.
#
# NOTE: --confirm_run_unsafe_code executes model-generated code in this shell.
# Correct on a disposable vast box, never on a workstation.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
export HF_ALLOW_CODE_EVAL=1

LOG=logs/eval/no_approx_$(date -u +%Y%m%dT%H%M%S).log
UTIL=0.75
MMLU12=mmlu_abstract_algebra,mmlu_business_ethics,mmlu_college_computer_science,mmlu_computer_security,mmlu_econometrics,mmlu_high_school_biology,mmlu_high_school_us_history,mmlu_machine_learning,mmlu_philosophy,mmlu_professional_medicine,mmlu_sociology,mmlu_world_religions

mkdir -p logs/eval
echo "=== no_approx eval $(date -u) ===" | tee "$LOG"

# Guard on the ARTIFACT, never on the directory (empty dirs from killed jobs
# made four eval arms silently vanish on 2026-08-01).
result_exists() { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }

# vLLM hangs AFTER lm_eval writes results, so a plain sequential call never
# returns. Poll for the artifact, then kill ONLY our own child EngineCore.
# Never `pkill -f` a global pattern -- that killed three concurrent jobs once.
run_lm () {   # $1=outdir $2=model $3=tasks $4=fewshot
  local out="$1" md="$2" tasks="$3" shots="$4"
  result_exists "$out" && { echo "[skip] $out" | tee -a "$LOG"; return 0; }
  echo "--- $(date -u +%H:%M:%S) $out ---" | tee -a "$LOG"
  lm_eval --model vllm \
    --model_args "pretrained=${md},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=${UTIL}" \
    --tasks "$tasks" --num_fewshot "$shots" --batch_size auto \
    --confirm_run_unsafe_code --output_path "$out" >>"$LOG" 2>&1 &
  local pid=$! waited=0
  while kill -0 "$pid" 2>/dev/null; do
    if result_exists "$out"; then
      sleep 5
      local ep; ep="$(pgrep -P "$pid" -f 'VLLM::EngineCore' | head -1 || true)"
      [ -n "$ep" ] && { echo "[cleanup] EngineCore $ep" | tee -a "$LOG"; kill -TERM "$ep" 2>/dev/null || true; }
      sleep 10; kill -TERM "$pid" 2>/dev/null || true
      break
    fi
    sleep 10; waited=$((waited+10))
    [ "$waited" -gt 2400 ] && { echo "[TIMEOUT] $out" | tee -a "$LOG"; kill -TERM "$pid" 2>/dev/null; break; }
  done
  wait "$pid" 2>/dev/null || true
  sleep 5
  result_exists "$out" || echo "[FAIL] $out produced no results" | tee -a "$LOG"
}

run_safety () {   # $1=tag $2=model
  local tag="$1" md="$2"
  if [ -f "results/${tag}/generations.jsonl" ]; then
    echo "[skip] safety $tag" | tee -a "$LOG"
  else
    echo "--- $(date -u +%H:%M:%S) safety $tag ---" | tee -a "$LOG"
    python -u experiments/p0_baseline_eval.py \
      --run-id "$tag" --model-id "$md" \
      --prompt-source advbench --advbench-source walledai --advbench-split train \
      --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
      --max-new-tokens 512 --max-length 4096 \
      --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
      --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 >>"$LOG" 2>&1
  fi
  # judging is OpenRouter API only -- no GPU
  [ -f "results/${tag}_judged/summary.json" ] || \
    python -u experiments/judge_generations.py \
      --generations "results/${tag}/generations.jsonl" --run-id "${tag}_judged" \
      --num-workers 32 >>"$LOG" 2>&1
}

# ---- the two real heretic checkpoints: full battery ----
for T in t24 t63; do
  D="outputs/heretic_no_approx_lvb_s1_${T}"
  TAG="noapx_s1_${T}"
  [ -f "$D/model.safetensors" ] || { echo "[MISSING] $D" | tee -a "$LOG"; continue; }
  run_safety "$TAG" "$D"
  run_lm "results/${TAG}_arc"       "$D" arc_challenge 0
  run_lm "results/${TAG}_mmlu"      "$D" "$MMLU12"     5
  run_lm "results/${TAG}_gsm8k"     "$D" gsm8k         5
  run_lm "results/${TAG}_humaneval" "$D" humaneval     0
  run_lm "results/${TAG}_mbpp"      "$D" mbpp          3
done

# ---- reference cells: code benchmarks only (arc/mmlu/gsm8k already exist) ----
run_lm "results/code_lvb_clean_humaneval" outputs/lvb_clean humaneval 0
run_lm "results/code_lvb_clean_mbpp"      outputs/lvb_clean mbpp      3
run_lm "results/code_lbase_humaneval" meta-llama/Llama-3.2-1B-Instruct humaneval 0
run_lm "results/code_lbase_mbpp"      meta-llama/Llama-3.2-1B-Instruct mbpp      3

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
echo "log: $LOG"
