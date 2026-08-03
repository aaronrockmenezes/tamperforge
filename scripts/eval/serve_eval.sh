#!/usr/bin/env bash
# Full eval battery against ONE persistent vLLM server per model.
#
# The old chain started a fresh in-process vLLM engine for every eval -- advbench,
# xstest x2, arc, mmlu, gsm8k, humaneval, mbpp = 8 startups per arm at ~75s of load +
# CUDA-graph compile each. ~10min of pure overhead against ~20min of wall clock. This
# starts the server once per model and points everything at the HTTP endpoint.
#
# Usage:  bash serve_eval.sh <tag> <model-dir> <thinking:off|default>
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
# MAXLEN 8192 not 4096: the completions endpoint rejects prompt_tokens + max_tokens >
# max_model_len with HTTP 400, where IN-PROCESS lm_eval handles the overflow itself. MBPP
# (3-shot, longest prompts in the battery) 400s at 4096; nothing else does. Same prompts and
# few-shot either way, so scores are unaffected -- it only permits longer sequences.
#
# NOT 8000: caddy (the vast.ai instance web server) holds that port permanently, so
# vllm serve dies with OSError EADDRINUSE and the harness silently reports nothing ran.
PORT="${PORT:-8765}"
UTIL="${UTIL:-0.85}"
LOG=logs/eval/serve_${TAG}_$(date -u +%Y%m%dT%H%M%S).log
PDIR=scripts/external_benches/prompts
MMLU12=mmlu_abstract_algebra,mmlu_business_ethics,mmlu_college_computer_science,mmlu_computer_security,mmlu_econometrics,mmlu_high_school_biology,mmlu_high_school_us_history,mmlu_machine_learning,mmlu_philosophy,mmlu_professional_medicine,mmlu_sociology,mmlu_world_religions
mkdir -p logs/eval

say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
have () { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }

[ -f "$MD/model.safetensors" ] || { say "[MISSING] $MD"; exit 1; }

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
  --gpu-memory-utilization "$UTIL" --max-model-len ${MAXLEN:-8192} --dtype bfloat16 \
  > "logs/eval/vllm_server_${TAG}.log" 2>&1 &
SERVER_PID=$!
say "  server pid $SERVER_PID, waiting for /health..."

# /health alone is NOT proof this is our server -- verify it advertises OUR tag.
ready=0
for i in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"${TAG}\""; then ready=1; break; fi
  kill -0 "$SERVER_PID" 2>/dev/null || { say "  [FAIL] server died during startup"; tail -20 "logs/eval/vllm_server_${TAG}.log" | tee -a "$LOG"; exit 1; }
  sleep 5
done
[ "$ready" = "1" ] || { say "  [FAIL] server not healthy after 450s"; kill -TERM "$SERVER_PID" 2>/dev/null; exit 1; }
say "  server ready"

# scoped teardown -- never a global pkill; kill OUR server and only its own children
cleanup () {
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

# --- generative evals: advbench + xstest, then judge (API-only, no GPU) --------------
gen () {   # $1=run-id  $2=prompt-args...
  local rid="$1"; shift
  if [ -f "results/${rid}/generations.jsonl" ]; then say "  [skip] gen $rid"; else
    say "  gen $rid"
    python -u experiments/gen_via_api.py --run-id "$rid" --served-model "$TAG" \
      --base-url "$BASE" --qwen-thinking "$TH" --num-workers 32 "$@" >>"$LOG" 2>&1
    [ -f "results/${rid}/generations.jsonl" ] || say "  [FAIL] gen $rid"
  fi
  [ -f "results/${rid}_judged/summary.json" ] && { say "  [skip] judge $rid"; return 0; }
  [ -f "results/${rid}/generations.jsonl" ] || return 0
  say "  judge $rid"
  python -u experiments/judge_generations.py --generations "results/${rid}/generations.jsonl" \
    --run-id "${rid}_judged" --num-workers 32 >>"$LOG" 2>&1
  [ -f "results/${rid}_judged/summary.json" ] || say "  [FAIL] judge $rid"
}

gen "$TAG"             --prompt-source advbench
gen "${TAG}_xssafe"    --prompt-file "$PDIR/xstest_safe.jsonl"
gen "${TAG}_xsunsafe"  --prompt-file "$PDIR/xstest_unsafe.jsonl"

# --- lm_eval over the same server ----------------------------------------------------
MA="model=${TAG},base_url=${BASE}/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,tokenizer=${MD}"
run_lm () {   # $1=outdir $2=tasks $3=fewshot
  have "$1" && { say "  [skip] $1"; return 0; }
  say "  lm_eval $1"
  lm_eval --model local-completions --model_args "$MA" \
    --tasks "$2" --num_fewshot "$3" --batch_size 1 \
    --confirm_run_unsafe_code --output_path "$1" >>"$LOG" 2>&1
  have "$1" || say "  [FAIL] $1"
}

run_lm "results/${TAG}_gsm8k"     gsm8k         5
run_lm "results/${TAG}_humaneval" humaneval     0
run_lm "results/${TAG}_mbpp"      mbpp          3
# ARC/MMLU are LOGLIKELIHOOD (echo+logprobs). SERVE_LL=api runs them over the server;
# SERVE_LL=inprocess stops the server first and uses the in-process engine -- the safe
# fallback when the smoke gate cannot reproduce known numbers over the API.
if [ "${SERVE_LL:-api}" = "api" ]; then
  run_lm "results/${TAG}_arc"       arc_challenge 0
  run_lm "results/${TAG}_mmlu"      "$MMLU12"     5
else
  say "  SERVE_LL=inprocess -- stopping server, running arc/mmlu in-process"
  cleanup; trap - EXIT
  for spec in "arc:arc_challenge:0" "mmlu:${MMLU12}:5"; do
    nm="${spec%%:*}"; rest="${spec#*:}"; tk="${rest%:*}"; sh="${rest##*:}"
    out="results/${TAG}_${nm}"
    have "$out" && { say "  [skip] $out"; continue; }
    say "  lm_eval(in-process) $out"
    lm_eval --model vllm \
      --model_args "pretrained=${MD},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45" \
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
fi

say "=== serve_eval $TAG DONE ==="
