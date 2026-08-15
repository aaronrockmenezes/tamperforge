#!/usr/bin/env bash
# Full eval battery against ONE persistent vLLM server per model.
#
# The old chain started a fresh in-process vLLM engine for every eval -- advbench,
# xstest x2, arc, mmlu, gsm8k, humaneval, mbpp = 8 startups per arm at ~75s of load +
# CUDA-graph compile each. ~10min of pure overhead against ~20min of wall clock. This
# starts the server once per model and points everything at the HTTP endpoint.
#
# Usage:  bash serve_eval.sh <tag> <model-dir> <thinking:off|default>
#         EVAL_PROFILE=advbench bash serve_eval.sh ...  # generation + judge only
#
# NOTE on task types: gsm8k / humaneval / mbpp are GENERATIVE and go over the API
# cleanly. arc_challenge and mmlu are LOGLIKELIHOOD -- they need echo+logprobs from the
# completions endpoint, which is the finickiest part of this. Smoke-test ARC before
# trusting a whole chain to it (scripts/eval/serve_eval_smoke.sh).
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
export HF_ALLOW_CODE_EVAL=1

TAG="${1:?usage: serve_eval.sh <tag> <model-dir> <thinking>}"
MD="${2:?}"
TH="${3:-off}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
case "$EVAL_PROFILE" in
  full|advbench) ;;
  *) echo "unknown EVAL_PROFILE=$EVAL_PROFILE (want full|advbench)" >&2; exit 2 ;;
esac
# MAXLEN 8192 not 4096: the completions endpoint rejects prompt_tokens + max_tokens >
# max_model_len with HTTP 400, where IN-PROCESS lm_eval handles the overflow itself. MBPP
# (3-shot, longest prompts in the battery) 400s at 4096; nothing else does. Same prompts and
# few-shot either way, so scores are unaffected -- it only permits longer sequences.
#
# NOT 8000: caddy (the vast.ai instance web server) holds that port permanently, so
# vllm serve dies with OSError EADDRINUSE and the harness silently reports nothing ran.
PORT="${PORT:-8765}"
UTIL="${UTIL:-0.85}"
# Concurrency. GEN_WORKERS hits the LOCAL vLLM server, so it is bounded by what vLLM reports it
# can schedule (~118 concurrent seqs for gemma-1b at 8k ctx on a 3090); 64 leaves headroom.
# JUDGE_WORKERS hits OpenRouter, which has no per-key worker cap, so it is bounded by rate limits
# rather than by us. Judging 520 rows at 32 workers was the slowest step in the chain.
GEN_WORKERS="${GEN_WORKERS:-64}"
JUDGE_WORKERS="${JUDGE_WORKERS:-48}"
# 512 not the 256 default. At 256 the judge TRUNCATES its own reply mid-"reason" -- measured
# 78/520 (15.0%) on gemma31bit_jitter_clean, finish_reason length/error, which trips the
# JUDGE_MAX_PARSE_FAIL_FRAC guard and refuses to write a summary. A passing judgment costs ~133
# completion tokens, so 512 has real headroom while staying cheap.
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-512}"
JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-90}"
SAFETY_MAX_NEW_TOKENS="${SAFETY_MAX_NEW_TOKENS:-512}"

# CONTEXT AND GENERATION BUDGET. Both were silently too small and both suppress scores rather
# than erroring, which is the worst way for a benchmark to be wrong.
#
#   EVAL_CTX   serves vLLM AND is passed to lm_eval as max_length. lm_eval's local-completions
#              backend defaults to 2048 -- while MMLU 5-shot runs ~2299 tokens, so every MMLU
#              number in this repo has been measured on truncated prompts (816 truncations were
#              logged in the E1/E2 runs). That is the standing "MMLU is understated repo-wide"
#              caveat, and it is this line.
#   EVAL_GEN   lm_eval's max_gen_toks defaults to 256. A model that thinks out loud, or that
#              becomes token-inefficient under the defence, gets scored on a cut-off answer --
#              indistinguishable from getting it wrong. Matters most for gsm8k/humaneval/mbpp
#              and for any reasoning-style model (phi-4-mini, Qwen3 thinking).
#
# 8k is the campaign-wide ceiling. Models with a smaller native window are capped further by
# model_context_len.py; lower caller/model defaults are left alone.
#
# Historical comparability boundary: generation was raised to 4k on 2026-08-14, but nothing
# predating that change has been re-measured at
# that budget; context remains capped at 8k.
# ceiling: every eval in results/ predating this was produced at 8k server ctx with lm_eval
# silently defaulting to max_length=2048 and max_gen_toks=256, so (a) all MMLU numbers are
# floors, not measurements, and (b) capability numbers for any verbose or reasoning-style model
# conflate "truncated" with "wrong". Cross-run capability comparisons that straddle this date
# are invalid. The gemma rrcenter/jitter arms are both at 8k so they stay comparable to EACH
# OTHER and to nothing else. upgrade: re-run the capability battery on the models still being
# quoted (Qwen version_G, Llama version_G, gemma base) before any capability number appears in
# a write-up, and record the ctx/gen budget alongside every future result.
EVAL_CTX="${EVAL_CTX:-8192}"
EVAL_GEN="${EVAL_GEN:-4096}"
LOG_ROOT="${LOG_ROOT:-logs/eval}"
LOG="${LOG_ROOT}/serve_${TAG}_$(date -u +%Y%m%dT%H%M%S).log"
PDIR=scripts/external_benches/prompts
MMLU12=mmlu_abstract_algebra,mmlu_business_ethics,mmlu_college_computer_science,mmlu_computer_security,mmlu_econometrics,mmlu_high_school_biology,mmlu_high_school_us_history,mmlu_machine_learning,mmlu_philosophy,mmlu_professional_medicine,mmlu_sociology,mmlu_world_religions
mkdir -p "$LOG_ROOT/vllm"

say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
have () { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }
model_ready () {
  [ -s "$1/model.safetensors" ] || [ -s "$1/model.safetensors.index.json" ] ||
    find "$1" -maxdepth 1 -type f -name 'model-*.safetensors' -size +0 -print -quit 2>/dev/null | grep -q .
}

REQUESTED_CTX="${MAXLEN:-$EVAL_CTX}"
EVAL_CTX=$(python scripts/tools/model_context_len.py "$MD" "$REQUESTED_CTX") || exit 1
[ "$EVAL_CTX" = "$REQUESTED_CTX" ] || say "  context capped: requested $REQUESTED_CTX, model supports $EVAL_CTX"

if [ -d "$MD" ]; then
  model_ready "$MD" || { say "[MISSING] model weights in $MD"; exit 1; }
else
  # An untouched Hub model is a valid base-control target. vLLM and the tokenizer
  # resolve it through the shared HF cache; trained/materialised arms remain local dirs.
  say "  using Hub model $MD"
fi

# --- reap orphans, then start ONE server -------------------------------------------
# match BOTH the engine and the API-server process: the old pattern only caught
# 'VLLM::EngineCore' and missed the api server, which is what held the port.
for p in $(pgrep -f 'VLLM::EngineCore|vllm serve' 2>/dev/null); do
  [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = "1" ] && { say "[reap] orphan $p"; kill -9 "$p"; }
done
sleep 3
# never start on an occupied port -- vllm dies with EADDRINUSE and /health then answers from
# whatever else is listening, which is how an arm silently got another model's server.
if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  say "  [FAIL] port ${PORT} already in use; refusing to start (would talk to the wrong server)"
  ss -tlnp 2>/dev/null | grep ":${PORT} " | tee -a "$LOG"
  exit 1
fi

say "=== serve_eval $TAG ($MD) ==="
vllm serve "$MD" --served-model-name "$TAG" --port "$PORT" \
  --gpu-memory-utilization "$UTIL" --max-model-len "$EVAL_CTX" --dtype bfloat16 \
  > "$LOG_ROOT/vllm/vllm_server_${TAG}.log" 2>&1 &
SERVER_PID=$!
say "  server pid $SERVER_PID, waiting for /health..."

# /health alone is NOT proof this is our server -- verify it advertises OUR tag.
ready=0
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"${TAG}\""; then ready=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { say "  [FAIL] server died during startup"; tail -20 "$LOG_ROOT/vllm/vllm_server_${TAG}.log" | tee -a "$LOG"; exit 1; }
  sleep 5
done
[ "$ready" = "1" ] || { say "  [FAIL] server not healthy after 450s"; kill -TERM "$SERVER_PID" 2>/dev/null; exit 1; }
say "  server ready"

# scoped teardown -- never a global pkill; kill OUR server and only its own children
cleanup () {
  if [ -n "${JUDGE_PID:-}" ] && kill -0 "$JUDGE_PID" 2>/dev/null; then
    say "  stopping async judge $JUDGE_PID"
    kill -TERM "$JUDGE_PID" 2>/dev/null || true
    wait "$JUDGE_PID" 2>/dev/null || true
  fi
  say "  stopping server $SERVER_PID"
  for c in $(pgrep -P "$SERVER_PID" 2>/dev/null); do kill -TERM "$c" 2>/dev/null || true; done
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  sleep 8
  kill -9 "$SERVER_PID" 2>/dev/null || true
  for p in $(pgrep -f 'VLLM::EngineCore' 2>/dev/null); do
    [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = "1" ] && kill -9 "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT

BASE="http://127.0.0.1:${PORT}/v1"
JUDGE_PID=""
JUDGE_RIDS=()

# --- generative evals: advbench + xstest ---------------------------------------------
gen () {   # $1=run-id  $2=prompt-args...
  local rid="$1"; shift
  if [ -f "results/${rid}/generations.jsonl" ]; then say "  [skip] gen $rid"; else
    say "  gen $rid"
    if ! python -u experiments/gen_via_api.py --run-id "$rid" --served-model "$TAG" \
      --base-url "$BASE" --qwen-thinking "$TH" --num-workers "$GEN_WORKERS" "$@" >>"$LOG" 2>&1
    then
      say "  [FAIL] generation process $rid"
      return 1
    fi
    [ -f "results/${rid}/generations.jsonl" ] || {
      say "  [FAIL] generations missing $rid"
      return 1
    }
  fi
  if [ -f "results/.defer_api_scoring" ]; then
    say "  [defer] judge $rid (OpenRouter quota marker present)"
    return 0
  fi
  [ -f "results/${rid}_judged/summary.json" ] && { say "  [skip] judge $rid"; return 0; }
  [ -f "results/${rid}/generations.jsonl" ] || return 0
  JUDGE_RIDS+=("$rid")
}

judge_queue () {
  local rid failed=0
  for rid in "${JUDGE_RIDS[@]}"; do
    say "  async judge $rid"
    if ! python -u experiments/judge_generations.py \
      --generations "results/${rid}/generations.jsonl" \
      --run-id "${rid}_judged" --num-workers "$JUDGE_WORKERS" \
      --judge-max-tokens "$JUDGE_MAX_TOKENS" \
      --judge-timeout-seconds "$JUDGE_TIMEOUT_SECONDS" >>"$LOG" 2>&1; then
      say "  [FAIL] judge process $rid"
      failed=1
    elif [ ! -f "results/${rid}_judged/summary.json" ]; then
      say "  [FAIL] judge summary $rid"
      failed=1
    fi
  done
  return "$failed"
}

start_judging () {
  [ "${#JUDGE_RIDS[@]}" -gt 0 ] || return 0
  judge_queue &
  JUDGE_PID=$!
  say "  async judge queue pid $JUDGE_PID (${#JUDGE_RIDS[@]} runs)"
}

await_judging () {
  [ -n "$JUDGE_PID" ] || return 0
  local pid="$JUDGE_PID"
  JUDGE_PID=""
  if ! wait "$pid"; then
    say "  [FAIL] async judge queue"
    return 1
  fi
}

gen "$TAG"             --prompt-source advbench --max-new-tokens "$SAFETY_MAX_NEW_TOKENS" || exit 1
[ "$EVAL_PROFILE" = "advbench" ] && {
  start_judging
  await_judging || exit 1
  say "=== serve_eval $TAG DONE (AdvBench only) ==="
  exit 0
}
gen "${TAG}_xssafe"    --prompt-file "$PDIR/xstest_safe.jsonl" || exit 1
gen "${TAG}_xsunsafe"  --prompt-file "$PDIR/xstest_unsafe.jsonl" || exit 1
start_judging

# --- lm_eval over the same server ----------------------------------------------------
# Keep requests token-aware: lm-eval can then left-truncate prompt + generation to its
# max_length. With tokenized_requests=False it explicitly skips that check and SmolLM2 sent
# 8,193 tokens to an 8,192-token server.
MA="model=${TAG},base_url=${BASE}/completions,num_concurrent=${GEN_WORKERS},max_retries=3,tokenized_requests=True,tokenizer=${MD},max_length=${EVAL_CTX},max_gen_toks=${EVAL_GEN}"
run_lm () {   # $1=outdir $2=tasks $3=fewshot
  have "$1" && { say "  [skip] $1"; return 0; }
  say "  lm_eval $1"
  if ! lm_eval --model local-completions --model_args "$MA" \
    --tasks "$2" --num_fewshot "$3" --batch_size 1 \
    --gen_kwargs "max_gen_toks=${EVAL_GEN}" \
    --confirm_run_unsafe_code --output_path "$1" >>"$LOG" 2>&1
  then
    say "  [FAIL] lm_eval process $1"
    return 1
  fi
  have "$1" || { say "  [FAIL] lm_eval artifact $1"; return 1; }
}

run_lm "results/${TAG}_gsm8k"     gsm8k         5 || exit 1
run_lm "results/${TAG}_humaneval" humaneval     0 || exit 1
run_lm "results/${TAG}_mbpp"      mbpp          3 || exit 1
# ARC/MMLU are LOGLIKELIHOOD (echo+logprobs). SERVE_LL=api runs them over the server;
# SERVE_LL=inprocess stops the server first and uses the in-process engine -- the safe
# fallback when the smoke gate cannot reproduce known numbers over the API.
if [ "${SERVE_LL:-api}" = "api" ]; then
  run_lm "results/${TAG}_arc"       arc_challenge 0 || exit 1
  if [ "${SKIP_MMLU:-0}" = 1 ]; then
    say "  [skip requested] MMLU"
  else
    run_lm "results/${TAG}_mmlu"      "$MMLU12"     5 || exit 1
  fi
else
  say "  SERVE_LL=inprocess -- stopping server, running arc/mmlu in-process"
  cleanup; trap - EXIT
  SPECS=("arc:arc_challenge:0")
  [ "${SKIP_MMLU:-0}" = 1 ] || SPECS+=("mmlu:${MMLU12}:5")
  for spec in "${SPECS[@]}"; do
    nm="${spec%%:*}"; rest="${spec#*:}"; tk="${rest%:*}"; sh="${rest##*:}"
    out="results/${TAG}_${nm}"
    have "$out" && { say "  [skip] $out"; continue; }
    say "  lm_eval(in-process) $out"
    lm_eval --model vllm \
      --model_args "pretrained=${MD},dtype=bfloat16,trust_remote_code=True,max_model_len=${EVAL_CTX},gpu_memory_utilization=0.45" \
      --tasks "$tk" --num_fewshot "$sh" --batch_size auto \
      --confirm_run_unsafe_code --output_path "$out" >>"$LOG" 2>&1 &
    pid=$!; waited=0
    while kill -0 "$pid" 2>/dev/null; do
      if have "$out"; then
        sleep 5
        ep="$(pgrep -P "$pid" -f 'VLLM::EngineCore' | head -1 || true)"
        [ -n "$ep" ] && kill -TERM "$ep" 2>/dev/null || true
        sleep 10; kill -TERM "$pid" 2>/dev/null || true; break
      fi
      sleep 10; waited=$((waited+10))
      [ "$waited" -gt 2400 ] && { say "  [TIMEOUT] $out"; kill -TERM "$pid" 2>/dev/null; break; }
    done
    wait "$pid" 2>/dev/null || true; sleep 5
    have "$out" || say "  [FAIL] $out"
  done
  [ "${SKIP_MMLU:-0}" != 1 ] || say "  [skip requested] MMLU"
fi

await_judging || exit 1
say "=== serve_eval $TAG DONE ==="
