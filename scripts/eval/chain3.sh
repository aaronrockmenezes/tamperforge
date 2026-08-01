#!/usr/bin/env bash
# Stage 3: code-generation capability (humaneval + mbpp).
#
# Why these: ARC/MMLU/IFEval barely move under attack because ranking 4 short
# continuations (or satisfying a format constraint) survives semantic collapse.
# Only GSM8K craters. Code generation is the same kind of measurement as GSM8K
# -- long coherent output, all-or-nothing scoring -- so it should track the
# real capability loss, and a degeneration loop scores exactly 0.
#
# Waits for chain2 to finish rather than running concurrently: two vLLM
# evaluators on one GPU is what wedged the box last time.
#
# NOTE: --confirm_run_unsafe_code executes model-generated code in this shell.
# Correct on a disposable vast box, never on a workstation.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
export HF_ALLOW_CODE_EVAL=1

CHAIN2_LOG=logs/training_runs/final_chain2.log
LOG=logs/training_runs/final_chain3.log
UTIL=0.75

# Gate on chain2. Skip the wait with CHAIN3_NOWAIT=1 if chain2 is already done.
if [ "${CHAIN3_NOWAIT:-0}" != "1" ]; then
  echo "[wait] polling $CHAIN2_LOG for '=== DONE'"
  while ! grep -aq '^=== DONE' "$CHAIN2_LOG" 2>/dev/null; do sleep 60; done
  echo "[wait] chain2 finished, settling 60s"
  sleep 60
fi

echo "=== chain3 $(date -u) ===" | tee "$LOG"

result_exists() { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }
cleanup_leftovers() { pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true; sleep 5; }

# Same background+poll+kill-EngineCore pattern as chain2 (commit 1a4e603):
# vLLM hangs AFTER lm_eval writes results, so a plain sequential call never returns.
run_lm () {   # $1=outdir  $2=model dir  $3=tasks  $4=fewshot
  local out="$1" md="$2" tasks="$3" shots="$4"
  result_exists "$out" && { echo "[skip] $out" | tee -a "$LOG"; return 0; }
  echo "--- $out ---" | tee -a "$LOG"
  lm_eval --model vllm \
    --model_args "pretrained=${md},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=${UTIL}" \
    --tasks "$tasks" --num_fewshot "$shots" --batch_size auto \
    --confirm_run_unsafe_code --output_path "$out" >>"$LOG" 2>&1 &
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
  grep -aE "\|pass@1|\|exact_match" "$LOG" | tail -2
}

# Budget-ordered: ~$0.50 left, ~6 min/arm. If it dies partway the arms that
# matter are already done. t65 first (the low-direction-layer trial), then the
# version_B MAD claim, then the rest.
for D in heretic_vb_t65 xvb_clean xvb_rank1 xvb_surg_k16 \
         heretic_vb_t99 heretic_vb_t17 \
         xva_clean xva_rank1 xva_surg_k16 heretic_va_s500_t175; do
  [ -d "outputs/$D" ] || continue
  run_lm "results/code_${D}_humaneval" "outputs/${D}" humaneval 0
  run_lm "results/code_${D}_mbpp"      "outputs/${D}" mbpp 3
done
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
