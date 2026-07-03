#!/usr/bin/env bash
# TIER 1: generate on the 3 external prompt sets with OUR vLLM harness, judge with OUR
# DeepSeek judge (uniform, no heavy judge models). Fast first-pass on the 4090.
# Tier 2 (official per-bench judges: Llama3-70B / ft-Mistral-7B / Gemma-2B) = see README,
# run on the Blackwell box.
#
# Point at ANY model dir/id (materialize v7 first via save_p1b_checkpoint if needed):
#   MODEL=google/gemma-3-1b-it TAG=gemma_base CUDA_VISIBLE_DEVICES=0 bash scripts/external_benches/run_tier1.sh
#   MODEL=/workspace/outputs/gemma_v7_hf TAG=gemma_v7 CUDA_VISIBLE_DEVICES=0 bash scripts/external_benches/run_tier1.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="${PY:-python}"
MODEL="${MODEL:?set MODEL=<hf id or dir>}"
TAG="${TAG:?set TAG=<short name>}"
BENCHES="${BENCHES:-strongreject jailbreakbench sorrybench}"
PDIR="scripts/external_benches/prompts"

for b in $BENCHES; do
  pf="$PDIR/$b.jsonl"
  [[ -f "$pf" ]] || { echo "!! $pf missing — run fetch_prompts.py --only $b first"; exit 1; }
  echo "### tier1 [$TAG] $b ($(wc -l < "$pf") prompts) ###"
  $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$MODEL" \
    --prompt-file "$pf" --n-prompts -1 --max-new-tokens 512 --max-length 4096 \
    --vllm-batch-size 64 --n-arc 0 --run-id "ext_${b}_${TAG}"
done
echo "### done — judge results/ext_*_${TAG}/generations.jsonl locally (our DeepSeek judge) ###"
