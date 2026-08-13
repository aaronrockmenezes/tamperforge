#!/usr/bin/env bash
# Same-harness GSM8K control for untouched google/gemma-3-1b-it.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
export HF_ALLOW_CODE_EVAL=1

TAG=gbase_clean
OUT=results/gbase_clean_gsm8k
MODEL_ID=google/gemma-3-1b-it
PORT=8765
LOG=logs/eval/gemma_base_gsm8k_$(date -u +%Y%m%dT%H%M%S).log
VLLM_LOG=logs/eval/vllm/vllm_gbase_clean_gsm8k.log
mkdir -p logs/eval/vllm results

say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
have_result () { find "$OUT" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }

if have_result; then
  say "[skip] base GSM8K result already exists"
  exit 0
fi

if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  say "[FAIL] port ${PORT} is already occupied"
  exit 1
fi

say "starting base Gemma vLLM"
vllm serve "$MODEL_ID" --served-model-name "$TAG" --port "$PORT" \
  --gpu-memory-utilization 0.85 --max-model-len 8192 --dtype bfloat16 \
  > "$VLLM_LOG" 2>&1 &
SERVER_PID=$!

cleanup () {
  for child in $(pgrep -P "$SERVER_PID" 2>/dev/null); do kill -TERM "$child" 2>/dev/null || true; done
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  sleep 8
  kill -9 "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 90); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"${TAG}\""; then
    ready=1
    break
  fi
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 5
done
[ "$ready" = 1 ] || { say "[FAIL] base Gemma server did not become ready"; exit 1; }

say "running GSM8K 5-shot"
MODEL_ARGS="model=${TAG},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,tokenizer=${MODEL_ID}"
lm_eval --model local-completions --model_args "$MODEL_ARGS" \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 \
  --confirm_run_unsafe_code --output_path "$OUT" >> "$LOG" 2>&1

have_result || { say "[FAIL] no GSM8K results artifact"; exit 1; }
RESULT_FILE=$(find "$OUT" -type f -name 'results_*.json' | sort | tail -1)
SCORE=$(jq -r '.results.gsm8k["exact_match,strict-match"]' "$RESULT_FILE")
say "GSM8K exact_match_strict=$SCORE"
say "BASE_GSM8K_COMPLETE"
