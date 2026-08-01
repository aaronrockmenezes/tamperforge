#!/usr/bin/env bash
# Direction-layer sweep AT HERETIC'S OPERATING POINT, on version_C s500.
#
# The old dl_sweep.sh varied the read layer under a DIFFERENT attack: flat alpha across all
# layers, plain (non row-normalized) application, read+write scope. Heretic's winning trial
# t71 is per-projection tents, write-only, FULL row-normalized, band-limited. Comparing the
# two would confound the layer axis with the attack shape, so this holds t71 fixed in every
# respect and sweeps ONLY direction_index.
#
# CONTROL: DL 14.31 is t71 itself and must reproduce 0.3231 harmful. If it does not, the
# sweep is not measuring what it claims and nothing downstream is usable.
#
# Question: is the low band special (collapse cannot be put there), or was 14.5 incidental?
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/dl_sweep_vc.log
echo "=== version_C DL sweep at t71's operating point $(date -u) ===" | tee "$LOG"
echo "=== control: DL 14.31 == t71 == 0.3231 harmful / GSM8K 0.3859 ===" | tee -a "$LOG"

for DL in 8.0 11.0 14.31 17.0 20.0 23.0 26.0; do
  TAG="dlvc_$(echo $DL | tr -d .)"
  D="outputs/${TAG}"
  echo "########## direction_index=${DL} ##########" | tee -a "$LOG"
  [ -f "$D/model.safetensors" ] || python -u experiments/version_c_replay.py \
      --checkpoint outputs/version_c_qwen_500.pt --trial t71 \
      --params-json results/heretic_vc_trials.json \
      --direction-recipe heretic --application heretic_full \
      --override-direction-index "$DL" --out "$D" 2>&1 | tee -a "$LOG"
  [ -f "results/${TAG}/generations.jsonl" ] || python -u experiments/p0_baseline_eval.py \
    --run-id "${TAG}" --model-id "$D" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking off 2>&1 | tee -a "$LOG"
  [ -f "results/${TAG}_judged/summary.json" ] || python -u experiments/judge_generations.py \
    --generations "results/${TAG}/generations.jsonl" --run-id "${TAG}_judged" \
    --num-workers 32 2>&1 | tee -a "$LOG"
  o="results/dlvccap_$(echo $DL | tr -d .)_gsm8k"
  find "$o" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . || lm_eval --model vllm \
    --model_args "pretrained=${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45" \
    --tasks gsm8k --num_fewshot 5 --batch_size auto --output_path "$o" 2>&1 \
    | tee -a "$LOG" | grep -aE "\|gsm8k"
  rm -rf "$D"
done
echo "=== DL SWEEP DONE $(date -u) ===" | tee -a "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"
