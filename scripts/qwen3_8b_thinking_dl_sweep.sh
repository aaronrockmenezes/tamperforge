#!/usr/bin/env bash
# Direction-layer sweep for Qwen3-8B in thinking mode.
#
# Produces base-attacked dirs plus AdvBench-200 generations per candidate layer.
# Judge each run locally/with .env present; pick the layer with high coherent harm
# and non-collapsed generations as the ABL-v8 direction layer.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DLS="${DLS:-12 16 20 24 28 32}"
N_PROMPTS="${N_PROMPTS:-200}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
MAX_LENGTH="${MAX_LENGTH:-40960}"
BATCH="${BATCH:-16}"
GPU_MEM="${GPU_MEM:-0.82}"
JUDGE="${JUDGE:-1}"
RUN_TAG="${RUN_TAG:-fullthink_ctx40960_out32768}"
REUSE_OUTPUTS="${REUSE_OUTPUTS:-1}"

export TF_QWEN_THINKING=on
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for dl in $DLS; do
  out="outputs/qwen3_8b_thinking_base_att_L${dl}"
  run="qwen3_8b_thinking_base_att_L${dl}_adv${N_PROMPTS}_${RUN_TAG}"
  echo "############ DL=${dl} -> ${out} ############"
  if [ "$REUSE_OUTPUTS" = "1" ] && [ -f "$out/config.json" ]; then
    echo "[reuse] existing attacked model: $out"
  else
    "$PY" experiments/save_p1b_checkpoint.py \
      --model-id "$MODEL" \
      --attack all \
      --direction-layer "$dl" \
      --qwen-thinking on \
      --out "$out"
  fi

  "$PY" experiments/p0_baseline_eval.py \
    --backend vllm \
    --model-id "$out" \
    --prompt-source advbench \
    --advbench-source walledai \
    --n-prompts "$N_PROMPTS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-length "$MAX_LENGTH" \
    --vllm-batch-size "$BATCH" \
    --vllm-gpu-memory-utilization "$GPU_MEM" \
    --qwen-thinking on \
    --vllm-temperature 0.6 \
    --vllm-top-p 0.95 \
    --vllm-top-k 20 \
    --n-arc 0 \
    --run-id "$run"

  if [ "$JUDGE" = "1" ]; then
    "$PY" experiments/judge_generations.py \
      --generations "results/${run}/generations.jsonl" \
      --run-id "${run}_judged" \
      --num-workers "${JUDGE_WORKERS:-32}"
  fi
done

echo "### done. Inspect results/qwen3_8b_thinking_base_att_L*_adv${N_PROMPTS}_${RUN_TAG}_judged/summary.json and set DL=... ###"
