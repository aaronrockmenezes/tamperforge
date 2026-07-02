#!/usr/bin/env bash
# Abliteration battery for ONE ABL-v7 model: the actual multi-model result.
# 4 conditions x {safety AdvBench 520 @512 (gen only -> judge locally), capability ARC+MMLU}.
# Win = abliterating the TRAINED model -> ~0 ASR/gibberish + capability crater,
#       while abliterating the BASE   -> coherent harm + capability intact.
#
#   MID=meta-llama/Llama-3.2-1B-Instruct CKPT=outputs/tamper_resistant_llama32_1b_v7.pt \
#   DL=8 TAG=llama CUDA_VISIBLE_DEVICES=1 bash scripts/multimodel_abl_v7/run_battery.sh
#
# direction-layer (DL) per model: gemma 13, Llama 8, Qwen3(0.6b/1.7b) 14.

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PY:-python}"
MID="${MID:?set MID=<hf model id>}"
CKPT="${CKPT:?set CKPT=<trained .pt>}"
DL="${DL:?set DL=<direction-layer>}"
TAG="${TAG:?set TAG=<short name>}"
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"

echo "### [$TAG] battery: MID=$MID CKPT=$CKPT DL=$DL CUDA=$CUDA_VISIBLE_DEVICES ###"

# 1) materialize the 3 attacked/trained HF dirs (base-clean = MID directly, no materialize)
$PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$CKPT" --attack none \
    --direction-layer "$DL" --out "outputs/${TAG}_v7_clean"
$PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$CKPT" --attack all \
    --direction-layer "$DL" --out "outputs/${TAG}_v7_att"
$PY experiments/save_p1b_checkpoint.py --model-id "$MID" --attack all \
    --direction-layer "$DL" --out "outputs/${TAG}_base_att"

# 2) safety generations (AdvBench full 520 @512) — gen only; JUDGE LOCALLY after
saf() { $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
    --prompt-source advbench --advbench-source walledai --n-prompts 520 \
    --max-new-tokens 512 --max-length 4096 --vllm-batch-size 64 --n-arc 0 --run-id "$2"; }
saf "$MID"                       "${TAG}_base_clean_adv520"
saf "outputs/${TAG}_base_att"    "${TAG}_base_att_adv520"
saf "outputs/${TAG}_v7_clean"    "${TAG}_v7_clean_adv520"
saf "outputs/${TAG}_v7_att"      "${TAG}_v7_att_adv520"

# 3) capability (ARC full + MMLU 12-topic) on all 4 conditions
cap() { local path="$1" name="$2"
  lm_eval --model vllm --model_args "pretrained=${path},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9" \
    --tasks arc_challenge --num_fewshot 0 --batch_size auto --output_path "results/cap_${name}_arc"
  lm_eval --model vllm --model_args "pretrained=${path},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9" \
    --tasks "$MMLU" --num_fewshot 0 --batch_size auto --output_path "results/cap_${name}_mmlu"; }
cap "$MID"                    "${TAG}_base_clean"
cap "outputs/${TAG}_base_att" "${TAG}_base_att"
cap "outputs/${TAG}_v7_clean" "${TAG}_v7_clean"
cap "outputs/${TAG}_v7_att"   "${TAG}_v7_att"

# reclaim disk (keep generations + cap results)
rm -rf "outputs/${TAG}_v7_clean" "outputs/${TAG}_v7_att" "outputs/${TAG}_base_att"
echo "### [$TAG] battery done — judge ${TAG}_*_adv520 locally; cap in results/cap_${TAG}_* ###"
