#!/usr/bin/env bash
# GSM8K-only capability check for the Version G Qwen checkpoints trained on
# harmful AdvBench completions (full SFT and merged LoRA).
set -euo pipefail

cd /workspace/tamperforge
source /venv/main/bin/activate

mkdir -p logs/eval results
LOG=logs/eval/vg_harmful_ft_gsm8k.log
UTIL="${UTIL:-0.85}"

say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

have () {
  find "$1" -name 'results_*.json' -print -quit 2>/dev/null | grep -q .
}

reap_orphans () {
  local p
  for p in $(pgrep -f 'VLLM::EngineCore' 2>/dev/null || true); do
    if [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = "1" ]; then
      say "[reap] orphaned EngineCore $p"
      kill -9 "$p" 2>/dev/null || true
    fi
  done
}

run_gsm8k () { # tag model_dir
  local tag="$1" model_dir="$2"
  local out="results/${tag}_gsm8k"
  local pid waited=0 engine_pid=""

  [ -f "${model_dir}/model.safetensors" ] || {
    say "[FAIL] missing ${model_dir}/model.safetensors"
    return 1
  }
  if have "$out"; then
    say "[skip] existing result: $out"
    return 0
  fi

  reap_orphans
  say "=== GSM8K 5-shot full test: $tag ($model_dir) ==="
  lm_eval --model vllm \
    --model_args "pretrained=${model_dir},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=${UTIL}" \
    --tasks gsm8k --num_fewshot 5 --batch_size auto \
    --confirm_run_unsafe_code --output_path "$out" >>"$LOG" 2>&1 &
  pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    if have "$out"; then
      sleep 5
      engine_pid="$(pgrep -P "$pid" -f 'VLLM::EngineCore' | head -1 || true)"
      [ -n "$engine_pid" ] && kill -TERM "$engine_pid" 2>/dev/null || true
      sleep 10
      kill -TERM "$pid" 2>/dev/null || true
      break
    fi
    sleep 10
    waited=$((waited + 10))
    if [ "$waited" -gt 2400 ]; then
      say "[TIMEOUT] $out"
      kill -TERM "$pid" 2>/dev/null || true
      break
    fi
  done
  wait "$pid" 2>/dev/null || true
  sleep 5
  have "$out" || { say "[FAIL] no result artifact: $out"; return 1; }
}

report () {
  python - "$@" <<'PY' | tee -a "$LOG"
import glob, json, sys
for tag in sys.argv[1:]:
    paths = glob.glob(f"results/{tag}_gsm8k/**/results_*.json", recursive=True)
    if not paths:
        print(f"RESULT {tag}: MISSING")
        continue
    data = json.load(open(paths[0]))
    row = data["results"]["gsm8k"]
    strict = row.get("exact_match,strict-match")
    flex = row.get("exact_match,flexible-extract")
    print(f"RESULT {tag}: strict_match={strict:.6f} flexible_extract={flex:.6f} artifact={paths[0]}")
PY
}

run_gsm8k vg_harm_sft outputs/vg_harm_sft592
run_gsm8k vg_harm_lora outputs/vg_harm_lora_r16_592
report vg_harm_sft vg_harm_lora
say "=== Version G harmful-FT GSM8K complete ==="
