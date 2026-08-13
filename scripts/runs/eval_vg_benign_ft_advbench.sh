#!/usr/bin/env bash
# Focused AdvBench-520 + pinned-judge evaluation for the two benign post-training attacks.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/eval/vllm results

PORT=8765
SP=""
say () { echo "[$(date -u +%H:%M:%S)] $*"; }

serve () { # tag model-dir
  local tag="$1" md="$2"
  ss -tln 2>/dev/null | grep -q ":${PORT} " && { say "[FAIL] port ${PORT} busy"; return 1; }
  vllm serve "$md" --served-model-name "$tag" --port "$PORT" \
    --gpu-memory-utilization 0.85 --max-model-len 4096 --dtype bfloat16 \
    > "logs/eval/vllm/vllm_${tag}.log" 2>&1 &
  SP=$!
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"$tag\"" && return 0
    kill -0 "$SP" 2>/dev/null || { tail -20 "logs/eval/vllm/vllm_${tag}.log"; return 1; }
    sleep 5
  done
  say "[FAIL] server never advertised $tag"; return 1
}

unserve () {
  [ -n "$SP" ] || return 0
  for c in $(pgrep -P "$SP" 2>/dev/null || true); do kill -TERM "$c" 2>/dev/null || true; done
  kill -TERM "$SP" 2>/dev/null || true
  sleep 8
  kill -9 "$SP" 2>/dev/null || true
  SP=""
  sleep 5
}
trap unserve EXIT

eval_arm () { # tag model-dir
  local tag="$1" md="$2"
  [ -f "$md/model.safetensors" ] || { say "[FAIL] missing $md/model.safetensors"; exit 1; }
  if [ ! -f "results/${tag}/generations.jsonl" ]; then
    say "=== generate AdvBench-520: $tag ==="
    serve "$tag" "$md"
    python -u experiments/gen_via_api.py --run-id "$tag" --served-model "$tag" \
      --base-url "http://127.0.0.1:${PORT}/v1" --qwen-thinking off \
      --prompt-source advbench --num-workers 32
    unserve
  else
    say "[skip] generations exist: $tag"
  fi
  if [ ! -f "results/${tag}_judged/summary.json" ]; then
    say "=== judge AdvBench-520: $tag ==="
    python -u experiments/judge_generations.py \
      --generations "results/${tag}/generations.jsonl" \
      --run-id "${tag}_judged" --num-workers 32
  else
    say "[skip] judged summary exists: $tag"
  fi
  python - "$tag" <<'PY'
import json, sys
tag = sys.argv[1]
d = json.load(open(f"results/{tag}_judged/summary.json"))
print(f"RESULT {tag}: n={d['n']} harm={d['harmful_actionable_rate']:.6f} "
      f"gib={d['gibberish_rate']:.6f} refusal={d['judge_refusal_rate']:.6f} "
      f"asr={d['judge_asr']:.6f} parse_failures={d['parse_failures']}")
PY
}

eval_arm vg_benign_sft_adv outputs/vg_benign_sft1000
eval_arm vg_benign_lora_adv outputs/vg_benign_lora_r16_1000
say "=== ADVBench-only SFT + LoRA evaluation complete ==="
