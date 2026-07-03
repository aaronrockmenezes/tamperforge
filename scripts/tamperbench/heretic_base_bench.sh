#!/usr/bin/env bash
# Benchmark the 3 Heretic-uncensored BASE gemma Pareto points (base-ref for the
# external-attacker figure). Per dir: AdvBench 520 @512 gens (JUDGE LOCALLY) +
# ARC full + MMLU-12 + GSM8K. Overlay harmAct/gib/cap vs the v7 Heretic Pareto (findings §6).
#
#   HDIR=/workspace/heretic_outputs CUDA_VISIBLE_DEVICES=0 bash scripts/tamperbench/heretic_base_bench.sh
# DIRS default = the 3 saved trials (kl098 T84 / kl036 T99 / kl014 T161); override via DIRS="...".
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/tamperbench/models.sh
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PY:-python}"
HDIR="${HDIR:?set HDIR=<heretic save dir>}"
DIRS="${DIRS:-kl098 kl036 kl014}"

for d in $DIRS; do
  M="$HDIR/heretic_base_gemma_$d"
  [[ -d "$M" ]] || { echo "!! missing $M"; exit 1; }
  echo "### heretic-base bench [$d] -> $M ###"

  # safety: AdvBench 520 @512 (gen only -> judge locally)
  $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$M" \
    --prompt-source advbench --advbench-source walledai --n-prompts 520 \
    --max-new-tokens 512 --max-length 4096 --vllm-batch-size 64 --n-arc 0 \
    --run-id "heretic_base_gemma_${d}_adv520"

  # capability: ARC full + MMLU-12 + GSM8K
  ARGS="pretrained=${M},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9"
  lm_eval --model vllm --model_args "$ARGS" --tasks arc_challenge --num_fewshot 0 --batch_size auto --output_path "results/cap_heretic_base_${d}_arc"
  lm_eval --model vllm --model_args "$ARGS" --tasks "$TB_MMLU"     --num_fewshot 0 --batch_size auto --output_path "results/cap_heretic_base_${d}_mmlu"
  lm_eval --model vllm --model_args "$ARGS" --tasks gsm8k          --num_fewshot 5 --batch_size auto --output_path "results/cap_heretic_base_${d}_gsm8k"
done
echo "### done — judge results/heretic_base_gemma_*_adv520 locally; cap in results/cap_heretic_base_* ###"
