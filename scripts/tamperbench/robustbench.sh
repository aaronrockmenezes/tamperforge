#!/usr/bin/env bash
# ROBUSTBENCH — attack-robustness bench across all 6 models (3 base + 3 forged, ATTACKED).
# The core MAD result: abliterating the FORGED model yields ~0 coherent harm + gibberish
# + capability crater, while abliterating the BASE yields coherent harm + capability intact.
# Attacks per model:
#   A1  rank-1 all-layer abliteration (diffmeans @ DL)      — base + forged
#   A2  PER-LAYER adaptive abliteration (indep dir/layer)   — base + forged   [--per-layer]
# Safety = AdvBench 520 @512 (JUDGE LOCALLY). Capability = ARC + MMLU-12 on each attacked model.
# Optional off-distribution (HarmBench 200 + BeaverTails 1483) on forged rank-1: OFFDIST=1.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/tamperbench/robustbench.sh
#   ONLY=llama OFFDIST=1 CUDA_VISIBLE_DEVICES=0 bash scripts/tamperbench/robustbench.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/tamperbench/models.sh
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PY:-python}"
ONLY="${ONLY:-}"
OFFDIST="${OFFDIST:-0}"

saf() { $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
  --prompt-source "${3:-advbench}" --advbench-source walledai \
  --n-prompts "${4:-520}" --max-new-tokens 512 --max-length 4096 \
  --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
cap() { local path="$1" name="$2"
  lm_eval --model vllm --model_args "pretrained=${path},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9" \
    --tasks arc_challenge --num_fewshot 0 --batch_size auto --output_path "results/rb_cap_${name}_arc"
  lm_eval --model vllm --model_args "pretrained=${path},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9" \
    --tasks "$TB_MMLU" --num_fewshot 0 --batch_size auto --output_path "results/rb_cap_${name}_mmlu"; }

for row in "${TB_MODELS[@]}"; do
  read -r TAG MID CKPT DL <<< "$row"
  [[ -n "$ONLY" && "$ONLY" != "$TAG" ]] && continue
  echo "### ROBUSTBENCH [$TAG] DL=$DL ###"
  [[ -f "$CKPT" ]] || { echo "!! missing $CKPT — run pull_ckpts.sh"; exit 1; }

  # A1 rank-1 all-layer abliteration
  $PY experiments/save_p1b_checkpoint.py --model-id "$MID"                 --attack all --direction-layer "$DL" --out "outputs/rb_${TAG}_base_att"
  $PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$CKPT" --attack all --direction-layer "$DL" --out "outputs/rb_${TAG}_v7_att"
  # A2 per-layer adaptive abliteration (independent diffmeans dir every layer)
  $PY experiments/save_p1b_checkpoint.py --model-id "$MID"                 --attack all --per-layer --out "outputs/rb_${TAG}_base_pl"
  $PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$CKPT" --attack all --per-layer --out "outputs/rb_${TAG}_v7_pl"

  for cond in base_att v7_att base_pl v7_pl; do
    saf "outputs/rb_${TAG}_${cond}" "rb_${TAG}_${cond}_adv520"
    cap "outputs/rb_${TAG}_${cond}" "${TAG}_${cond}"
  done

  if [[ "$OFFDIST" == "1" ]]; then
    saf "outputs/rb_${TAG}_v7_att" "rb_${TAG}_v7_att_harmbench" harmbench 200
    saf "outputs/rb_${TAG}_v7_att" "rb_${TAG}_v7_att_beavertails" beavertails 1483
  fi

  rm -rf "outputs/rb_${TAG}_base_att" "outputs/rb_${TAG}_v7_att" \
         "outputs/rb_${TAG}_base_pl"  "outputs/rb_${TAG}_v7_pl"
done
echo "### ROBUSTBENCH done — judge rb_*_adv520 (+ harmbench/beavertails) locally; cap in results/rb_cap_* ###"
echo "Win: base_att/base_pl -> high harmAct (cap intact); v7_att/v7_pl -> ~0 harmAct + gibberish + cap crater."
