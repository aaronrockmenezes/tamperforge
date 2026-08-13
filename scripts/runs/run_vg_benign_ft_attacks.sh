#!/usr/bin/env bash
# Benign post-training attacks on version_G Qwen.
#
# Both arms use the exact base-generated Alpaca demonstrations used for the
# measured version_B repair attack. Full SFT keeps that run's LR; LoRA uses the
# standard higher adapter LR. The attack succeeds if ordinary post-training
# restores harmful behavior under the held-out rank-1 edit without an equivalent
# clean capability collapse.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

N_ROWS="${N_ROWS:-1000}"
EPOCHS="${EPOCHS:-1}"
FULL_LR="${FULL_LR:-1e-5}"
LORA_LR="${LORA_LR:-2e-4}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
CKPT=outputs/version_g_qwen_500.pt
DEMOS=results/sft_demos_base/generations.jsonl
FULL=outputs/vg_benign_sft${N_ROWS}
LORA=outputs/vg_benign_lora_r${LORA_RANK}_${N_ROWS}
PORT=8765

say () { echo "[$(date -u +%H:%M:%S)] $*"; }
[ -f "$CKPT" ] || { say "[FAIL] missing $CKPT"; exit 1; }
[ -f "$DEMOS" ] || { say "[FAIL] missing $DEMOS"; exit 1; }
mkdir -p logs/training_runs logs/probes logs/eval/vllm results outputs

train_arm () { # method out lr
  local method="$1" out="$2" lr="$3"
  if [ -f "$out/model.safetensors" ]; then
    say "[skip] $out exists"
    return
  fi
  say "=== ${method} attack: ${N_ROWS} requested rows, ${EPOCHS} epoch(s), lr ${lr} ==="
  local extra=()
  if [ "$method" = lora ]; then
    extra=(--lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" --lora-dropout 0)
  fi
  python -u experiments/ft_attack.py --model-id Qwen/Qwen3-0.6B \
    --checkpoint "$CKPT" --demos "$DEMOS" --n-shots "$N_ROWS" \
    --ft-epochs "$EPOCHS" --ft-lr "$lr" --max-len 1024 \
    --method "$method" "${extra[@]}" --out "$out" \
    2>&1 | tee "logs/training_runs/$(basename "$out").log"
  [ -f "$out/model.safetensors" ] || { say "[FAIL] $method produced nothing"; exit 1; }
}

train_arm full "$FULL" "$FULL_LR"
train_arm lora "$LORA" "$LORA_LR"

for spec in "sft:$FULL" "lora:$LORA"; do
  nm="${spec%%:*}"; md="${spec#*:}"
  say "=== smoke5 $nm ==="
  python scripts/probes/smoke5.py "$md" --max-new 120 --modes nothink,default \
    2>&1 | tee "logs/probes/smoke5_vg_benign_${nm}.log"
done

# Apply the original held-out attack after post-training. This directly tests
# whether fine-tuning dissolved version_G's wall rather than merely changing its
# clean refusal rate.
for spec in "sft:$FULL" "lora:$LORA"; do
  nm="${spec%%:*}"; md="${spec#*:}"; abl="${md}_rank1"
  if [ -f "$abl/model.safetensors" ]; then
    say "[skip] $abl exists"
  else
    say "=== rank-1 after $nm ==="
    python -u experiments/v11_surgical_ablation.py --model-id "$md" \
      --direction-layer 20 --cap-rank 0 --out "$abl"
  fi
done

SP=""
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

gen_and_judge () { # tag model-dir prompt args...
  local tag="$1" md="$2"; shift 2
  if [ ! -f "results/${tag}/generations.jsonl" ]; then
    serve "$tag" "$md"
    python -u experiments/gen_via_api.py --run-id "$tag" --served-model "$tag" \
      --base-url "http://127.0.0.1:${PORT}/v1" --qwen-thinking off \
      --num-workers 32 "$@"
    unserve
  else
    say "[skip] gen $tag"
  fi
  case "$tag" in mtb_*) return ;; esac
  if [ ! -f "results/${tag}_judged/summary.json" ]; then
    python -u experiments/judge_generations.py \
      --generations "results/${tag}/generations.jsonl" \
      --run-id "${tag}_judged" --num-workers 32
  else
    say "[skip] judge $tag"
  fi
}

for spec in "sft:$FULL" "lora:$LORA"; do
  nm="${spec%%:*}"; md="${spec#*:}"
  gen_and_judge "mtb_vg_benign_${nm}" "$md" \
    --prompt-file scripts/external_benches/prompts/mtbench_t1.jsonl --max-new-tokens 768
  gen_and_judge "vg_benign_${nm}_adv" "$md" --prompt-source advbench
  gen_and_judge "vg_benign_${nm}_r1_adv" "${md}_rank1" --prompt-source advbench
done

say "=== pinned MT-Bench judge ==="
python -u experiments/mtbench_single.py \
  --tags mtb_xbase_clean mtb_vg mtb_vg_benign_sft mtb_vg_benign_lora

say "=== GSM8K clean vs rank-1 ==="
for spec in \
  "vg_benign_sft_clean:$FULL" "vg_benign_sft_rank1:${FULL}_rank1" \
  "vg_benign_lora_clean:$LORA" "vg_benign_lora_rank1:${LORA}_rank1"; do
  nm="${spec%%:*}"; md="${spec#*:}"
  if find "results/${nm}_gsm8k" -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; then
    say "[skip] ${nm}_gsm8k"
    continue
  fi
  serve "$nm" "$md"
  lm_eval --model local-completions \
    --model_args "model=${nm},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,tokenizer=${md}" \
    --tasks gsm8k --num_fewshot 5 --batch_size 1 --output_path "results/${nm}_gsm8k"
  unserve
done

say "=== VERSION_G BENIGN FULL-SFT + LORA ATTACKS DONE ==="
df -h /workspace | tail -1
