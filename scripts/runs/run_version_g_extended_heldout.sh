#!/usr/bin/env bash
# Version G held-out external suite, adapted from the v8-era eval matrices.
# One persistent vLLM server per model; generate every benchmark first, then
# judge with the pinned uniform Tier-1 judge. Official benchmark judges are a
# separate Tier-2 result and must not be conflated with these summaries.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

PORT="${PORT:-8765}"
UTIL="${UTIL:-0.85}"
MAXLEN="${MAXLEN:-4096}"
WORKERS="${WORKERS:-32}"
JUDGE_WORKERS="${JUDGE_WORKERS:-32}"
SUITE=data/heldout_vg_20260804
LOG=logs/eval/version_g_extended_heldout_$(date -u +%Y%m%dT%H%M%S).log
mkdir -p logs/eval logs/eval/vllm results

HARM_BENCHES=(harmbench jailbreakbench strongreject sorrybench beavertails)
ADAPT_BENCHES=(xstest_safe xstest_unsafe orbench)

# tag:model_dir. Base clean + rank-1 establish attack headroom; Version G has
# clean/rank-1/surgical/Heretic; Qwen additionally includes benign and harmful FT.
MODEL_SPECS=(
  qbase_clean:outputs/xbase_clean_hf
  qbase_rank1:outputs/xbase_rank1
  qg_clean:outputs/version_g_qwen_500_clean
  qg_rank1:outputs/vg_rank1
  qg_surg_k16:outputs/vg_surg_k16
  qg_her_t73:outputs/vg_her_s0_att
  qg_benign_sft:outputs/vg_benign_sft1000
  qg_benign_lora:outputs/vg_benign_lora_r16_1000
  qg_harm_sft:outputs/vg_harm_sft592
  qg_harm_lora:outputs/vg_harm_lora_r16_592
  lbase_clean:outputs/lbase_clean_hf
  lbase_rank1:outputs/lbase_rank1
  lg_clean:outputs/version_g_llama_500_clean
  lg_rank1:outputs/vgl_rank1
  lg_surg_k16:outputs/vgl_surg_k16
  lg_her_t138:outputs/vgl_her_s0_att
)
ADAPT_TAGS=(qg_benign_sft qg_benign_lora qg_harm_sft qg_harm_lora)

say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
expected () { wc -l < "$SUITE/$1.jsonl" | tr -d ' '; }
have_exact () {
  local rid="$1" bench="$2"
  local f="results/${rid}/generations.jsonl"
  [ -f "$f" ] || return 1
  local got want; got="$(wc -l < "$f" | tr -d ' ')"; want="$(expected "$bench")"
  [ "$got" = "$want" ] || { say "[FAIL] stale/incomplete $f: got $got expected $want"; exit 1; }
}
is_adaptation () {
  local x; for x in "${ADAPT_TAGS[@]}"; do [ "$1" = "$x" ] && return 0; done; return 1
}

[ -f "$SUITE/manifest.json" ] || { say "[FAIL] missing frozen suite manifest"; exit 1; }
for b in "${HARM_BENCHES[@]}" "${ADAPT_BENCHES[@]}"; do
  [ -s "$SUITE/$b.jsonl" ] || { say "[FAIL] missing/empty $SUITE/$b.jsonl"; exit 1; }
done
for spec in "${MODEL_SPECS[@]}"; do
  md="${spec#*:}"; [ -f "$md/model.safetensors" ] || { say "[FAIL] missing $md/model.safetensors"; exit 1; }
done

SP=""
cleanup () {
  [ -n "$SP" ] || return 0
  for c in $(pgrep -P "$SP" 2>/dev/null || true); do kill -TERM "$c" 2>/dev/null || true; done
  kill -TERM "$SP" 2>/dev/null || true
  sleep 8
  kill -9 "$SP" 2>/dev/null || true
  SP=""
  for p in $(pgrep -f 'VLLM::EngineCore' 2>/dev/null || true); do
    [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = "1" ] && kill -9 "$p" 2>/dev/null || true
  done
  sleep 3
}
trap cleanup EXIT

serve () { # tag model_dir
  local tag="$1" md="$2"
  if ss -tln 2>/dev/null | grep -q ":${PORT} "; then say "[FAIL] port $PORT busy"; exit 1; fi
  say "=== serve $tag ($md) ==="
  vllm serve "$md" --served-model-name "$tag" --port "$PORT" \
    --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" --dtype bfloat16 \
    > "logs/eval/vllm/vllm_ext_${tag}.log" 2>&1 &
  SP=$!
  local ready=0
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"${tag}\""; then ready=1; break; fi
    kill -0 "$SP" 2>/dev/null || { tail -30 "logs/eval/vllm/vllm_ext_${tag}.log" | tee -a "$LOG"; exit 1; }
    sleep 5
  done
  [ "$ready" = 1 ] || { say "[FAIL] server not ready: $tag"; exit 1; }
}

gen () { # tag bench
  local tag="$1" bench="$2"
  local rid="vgho_${tag}_${bench}"
  if have_exact "$rid" "$bench"; then say "[skip] gen $rid"; return; fi
  say "gen $rid n=$(expected "$bench")"
  python -u experiments/gen_via_api.py --run-id "$rid" --served-model "$tag" \
    --base-url "http://127.0.0.1:${PORT}/v1" --prompt-file "$SUITE/$bench.jsonl" \
    --qwen-thinking off --num-workers "$WORKERS" --max-new-tokens 512 >>"$LOG" 2>&1
  have_exact "$rid" "$bench" || { say "[FAIL] generation artifact $rid"; exit 1; }
}

say "=== phase 1: generate frozen held-out suite ==="
for spec in "${MODEL_SPECS[@]}"; do
  tag="${spec%%:*}"; md="${spec#*:}"
  serve "$tag" "$md"
  for b in "${HARM_BENCHES[@]}"; do gen "$tag" "$b"; done
  if is_adaptation "$tag"; then for b in "${ADAPT_BENCHES[@]}"; do gen "$tag" "$b"; done; fi
  cleanup
done

judge () { # tag bench
  local tag="$1" bench="$2"
  local rid="vgho_${tag}_${bench}"
  [ -f "results/${rid}_judged/summary.json" ] && { say "[skip] judge $rid"; return; }
  say "judge $rid"
  python -u experiments/judge_generations.py --generations "results/${rid}/generations.jsonl" \
    --run-id "${rid}_judged" --num-workers "$JUDGE_WORKERS" >>"$LOG" 2>&1
  [ -f "results/${rid}_judged/summary.json" ] || { say "[FAIL] judge $rid"; exit 1; }
}

say "=== phase 2: pinned uniform Tier-1 judging ==="
for spec in "${MODEL_SPECS[@]}"; do
  tag="${spec%%:*}"
  for b in "${HARM_BENCHES[@]}"; do judge "$tag" "$b"; done
  if is_adaptation "$tag"; then judge "$tag" xstest_unsafe; fi
done

say "=== phase 3: pinned over-refusal scoring for adapted Qwen arms ==="
for tag in "${ADAPT_TAGS[@]}"; do
  for b in xstest_safe orbench; do
    rid="vgho_${tag}_${b}"
    [ -f "results/${rid}_orscore/summary.json" ] && { say "[skip] orscore $rid"; continue; }
    python -u scripts/external_benches/score_overrefusal.py \
      --generations "results/${rid}/generations.jsonl" --run-id "$rid" \
      --num-workers "$JUDGE_WORKERS" >>"$LOG" 2>&1
    [ -f "results/${rid}_orscore/summary.json" ] || { say "[FAIL] orscore $rid"; exit 1; }
  done
done

say "=== Version G frozen extended suite complete ==="
