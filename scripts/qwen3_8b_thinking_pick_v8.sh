#!/usr/bin/env bash
# Materialize, evaluate, judge, and auto-pick Qwen3-8B thinking-mode ABL-v8 snapshots.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3-8B}"
DL="${DL:?set DL}"
STEM="${STEM:-outputs/tamper_resistant_qwen3_8b_thinking_v8.pt}"

export TF_QWEN_THINKING=on
export QWEN_THINKING=on
export TF_IFEVAL_MAX_NEW="${TF_IFEVAL_MAX_NEW:-512}"
export VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0.6}"
export VLLM_TOP_P="${VLLM_TOP_P:-0.95}"
export VLLM_TOP_K="${VLLM_TOP_K:-20}"
export VLLM_PRESENCE_PENALTY="${VLLM_PRESENCE_PENALTY:-0.0}"

MID="$MODEL" STEM="$STEM" DL="$DL" bash scripts/pick_v8_best.sh

STEM="$STEM" WORKERS="${WORKERS:-32}" bash scripts/auto_pick_v8.sh

echo "### auto-pick result: results/auto_pick_v8_result.json"
