#!/usr/bin/env bash
# Frozen held-out suite for Gemma base and Version G variants. Generation only;
# judge_gemma_vg_extended_async.py scores complete artifacts concurrently.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

PORT="${PORT:-8765}"; UTIL="${UTIL:-0.85}"; MAXLEN="${MAXLEN:-4096}"
WORKERS="${WORKERS:-32}"; SUITE=data/heldout_vg_20260804
LOG=logs/eval/gemma_vg_extended_$(date -u +%Y%m%dT%H%M%S).log
mkdir -p logs/eval/vllm results outputs

BENCHES=(harmbench jailbreakbench strongreject sorrybench beavertails xstest_safe xstest_unsafe orbench)
MODELS=(
  gbase_clean:google/gemma-3-1b-it
  gg_clean:outputs/version_g_gemma_500_clean
  gg_her:outputs/vgg_her_s0_att
)
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
expected () { wc -l < "$SUITE/$1.jsonl" | tr -d ' '; }
have_exact () {
  local f="results/$1/generations.jsonl" got want
  [ -f "$f" ] || return 1
  got=$(wc -l < "$f" | tr -d ' '); want=$(expected "$2")
  [ "$got" = "$want" ]
}

[ -s "$SUITE/manifest.json" ] || { say "[FAIL] frozen suite missing"; exit 1; }

SP=""
cleanup () {
  [ -n "$SP" ] || return 0
  for c in $(pgrep -P "$SP" 2>/dev/null || true); do kill -TERM "$c" 2>/dev/null || true; done
  kill -TERM "$SP" 2>/dev/null || true; sleep 8; kill -9 "$SP" 2>/dev/null || true
  SP=""; sleep 3
}
trap cleanup EXIT
serve () {
  local tag="$1" md="$2" ready=0
  ss -tln 2>/dev/null | grep -q ":${PORT} " && { say "[FAIL] port $PORT busy"; exit 1; }
  say "serve $tag ($md)"
  vllm serve "$md" --served-model-name "$tag" --port "$PORT" \
    --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" --dtype bfloat16 \
    >"logs/eval/vllm/vllm_gext_${tag}.log" 2>&1 & SP=$!
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"${tag}\"" && { ready=1; break; }
    kill -0 "$SP" 2>/dev/null || break; sleep 5
  done
  [ "$ready" = 1 ] || { say "[FAIL] server startup $tag"; exit 1; }
}
gen () {
  local tag="$1" bench="$2" rid="gext_${1}_${2}"
  have_exact "$rid" "$bench" && { say "[skip] $rid"; return; }
  say "gen $rid n=$(expected "$bench")"
  python -u experiments/gen_via_api.py --run-id "$rid" --served-model "$tag" \
    --base-url "http://127.0.0.1:${PORT}/v1" --prompt-file "$SUITE/$bench.jsonl" \
    --qwen-thinking off --num-workers "$WORKERS" --max-new-tokens 512 >>"$LOG" 2>&1
  have_exact "$rid" "$bench" || { say "[FAIL] incomplete $rid"; exit 1; }
}

for spec in "${MODELS[@]}"; do
  tag=${spec%%:*}; md=${spec#*:}
  [ "$md" = google/gemma-3-1b-it ] || [ -s "$md/model.safetensors" ] || { say "[FAIL] missing $md"; exit 1; }
  serve "$tag" "$md"
  for b in "${BENCHES[@]}"; do gen "$tag" "$b"; done
  cleanup
done
say "=== GEMMA EXTENDED GENERATION COMPLETE ==="
