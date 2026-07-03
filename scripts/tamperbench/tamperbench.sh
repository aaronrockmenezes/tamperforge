#!/usr/bin/env bash
# TAMPERBENCH — product-credibility bench across all 6 models (3 base + 3 forged, CLEAN).
# Question: does the forged (ABL-v7) model behave like a normal safe assistant when NOT
# attacked? i.e. refuses harmful (AdvBench) AND keeps capability (ARC/MMLU) ~= base.
# This is the "free product" claim. Attack robustness = robustbench.sh (separate).
#
# Per model: eval BASE-clean (MID directly) + FORGED-clean (materialize ckpt, attack none).
# Safety = AdvBench 520 @512 generations (JUDGE LOCALLY after). Capability = ARC + MMLU-12.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/tamperbench/tamperbench.sh            # all 3 families
#   ONLY=gemma CUDA_VISIBLE_DEVICES=0 bash scripts/tamperbench/tamperbench.sh # one family
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/tamperbench/models.sh
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PY:-python}"
ONLY="${ONLY:-}"

saf() { $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
  --prompt-source advbench --advbench-source walledai --n-prompts 520 \
  --max-new-tokens 512 --max-length 4096 --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
cap() { local path="$1" name="$2"
  lm_eval --model vllm --model_args "pretrained=${path},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9" \
    --tasks arc_challenge --num_fewshot 0 --batch_size auto --output_path "results/tb_cap_${name}_arc"
  lm_eval --model vllm --model_args "pretrained=${path},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9" \
    --tasks "$TB_MMLU" --num_fewshot 0 --batch_size auto --output_path "results/tb_cap_${name}_mmlu"; }

for row in "${TB_MODELS[@]}"; do
  read -r TAG MID CKPT DL <<< "$row"
  [[ -n "$ONLY" && "$ONLY" != "$TAG" ]] && continue
  echo "### TAMPERBENCH [$TAG] base+forged clean (DL=$DL) ###"
  [[ -f "$CKPT" ]] || { echo "!! missing $CKPT — run pull_ckpts.sh"; exit 1; }

  # BASE clean
  saf "$MID" "tb_${TAG}_base_clean_adv520"
  cap "$MID" "${TAG}_base_clean"

  # FORGED clean (materialize ckpt, no attack)
  $PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$CKPT" \
      --attack none --direction-layer "$DL" --out "outputs/tb_${TAG}_v7_clean"
  saf "outputs/tb_${TAG}_v7_clean" "tb_${TAG}_v7_clean_adv520"
  cap "outputs/tb_${TAG}_v7_clean" "${TAG}_v7_clean"
  rm -rf "outputs/tb_${TAG}_v7_clean"
done
echo "### TAMPERBENCH done — judge tb_*_adv520 locally; cap in results/tb_cap_* ###"
echo "Expect: forged refuses AdvBench like base (low harmAct) + cap ~= base (free product)."
