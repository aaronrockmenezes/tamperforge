#!/usr/bin/env bash
# Direct train-on-test attack characterization: harmful full SFT and LoRA on
# mined AdvBench completions, followed only by AdvBench-520 + pinned judging.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs logs/eval/vllm results outputs

CKPT=outputs/version_g_qwen_500.pt
DEMOS=data/harm_targets_qwen.json
N_SHOTS="${N_SHOTS:-592}"
EPOCHS="${EPOCHS:-1}"
FULL_LR="${FULL_LR:-1e-5}"
LORA_LR="${LORA_LR:-2e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
FULL=outputs/vg_harm_sft${N_SHOTS}
LORA=outputs/vg_harm_lora_r${LORA_RANK}_${N_SHOTS}
PORT=8765
SP=""

say () { echo "[$(date -u +%H:%M:%S)] $*"; }
[ -f "$CKPT" ] || { say "[FAIL] missing $CKPT"; exit 1; }
[ -f "$DEMOS" ] || { say "[FAIL] missing $DEMOS"; exit 1; }

train_arm () { # method out lr
  local method="$1" out="$2" lr="$3"; shift 3
  if [ -f "$out/model.safetensors" ]; then
    say "[skip] checkpoint exists: $out"
    return
  fi
  say "=== harmful $method: $N_SHOTS pairs, $EPOCHS epoch(s), lr $lr ==="
  python -u experiments/ft_attack.py --model-id Qwen/Qwen3-0.6B \
    --checkpoint "$CKPT" --demos "$DEMOS" --n-shots "$N_SHOTS" \
    --ft-epochs "$EPOCHS" --ft-lr "$lr" --max-len 1024 \
    --method "$method" "$@" --out "$out" \
    2>&1 | tee "logs/training_runs/$(basename "$out").log"
  [ -f "$out/model.safetensors" ] || { say "[FAIL] no checkpoint: $out"; exit 1; }
}

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
  if [ ! -f "results/${tag}/generations.jsonl" ]; then
    say "=== generate AdvBench-520: $tag ==="
    serve "$tag" "$md"
    python -u experiments/gen_via_api.py --run-id "$tag" --served-model "$tag" \
      --base-url "http://127.0.0.1:${PORT}/v1" --qwen-thinking off \
      --prompt-source advbench --num-workers 32
    unserve
  fi
  if [ ! -f "results/${tag}_judged/summary.json" ]; then
    say "=== judge AdvBench-520: $tag ==="
    python -u experiments/judge_generations.py \
      --generations "results/${tag}/generations.jsonl" \
      --run-id "${tag}_judged" --num-workers 32
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

train_arm full "$FULL" "$FULL_LR"
train_arm lora "$LORA" "$LORA_LR" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --lora-dropout 0
eval_arm vg_harm_sft_adv "$FULL"
eval_arm vg_harm_lora_adv "$LORA"
say "=== harmful SFT + LoRA AdvBench attack complete ==="
