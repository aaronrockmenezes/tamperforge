#!/usr/bin/env bash
# Fast vLLM eval of a P1b-A checkpoint: 3 conditions x {safety, ARC, generative-MMLU}.
# The P1b-A model is pure weights -> save_pretrained -> vLLM runs it batched.
#
#   bash scripts/eval_p1b_vllm.sh outputs/tamper_resistant_p1b_v7.pt p1b_v7 all
#     $1 = checkpoint   $2 = tag   $3 = attack scope (all|mlp), default all
#
# Judge the *_gen/generations.jsonl LOCALLY afterward (OpenRouter key is local).

set -euo pipefail
cd /workspace/tamperforge
git fetch origin -q && git checkout origin/main -- experiments src scripts docs

CKPT="${1:-outputs/tamper_resistant_p1b_v7.pt}"
TAG="${2:-p1b_v7}"
SCOPE="${3:-all}"

# materialize the 3 conditions as HF model dirs
python experiments/save_p1b_checkpoint.py --checkpoint "$CKPT" --attack none         --out "outputs/${TAG}_clean"
python experiments/save_p1b_checkpoint.py --checkpoint "$CKPT" --attack "$SCOPE"      --out "outputs/${TAG}_att"
python experiments/save_p1b_checkpoint.py                      --attack "$SCOPE"      --out "outputs/${TAG}_base_att"

for cond in clean att base_att; do
  dir="outputs/${TAG}_${cond}"
  echo "=== $cond ($dir) ==="
  # safety: FULL AdvBench 520 (vLLM batched)
  python experiments/p0_baseline_eval.py --backend vllm --model-id "$dir" \
    --advbench-source walledai --n-advbench 520 --n-arc 0 \
    --max-new-tokens 128 --max-length 4096 --run-id "${TAG}_${cond}_gen"
  # capability: ARC-Challenge 25-shot FULL (log-likelihood)
  lm_eval --model vllm \
    --model_args pretrained="$dir",dtype=bfloat16,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
    --tasks arc_challenge --num_fewshot 25 --batch_size 1 \
    --output_path "results/lm_eval_${TAG}_${cond}_arc25_full" --log_samples
  # capability: GENERATIVE MMLU (collapse signal — parse_rate)
  python experiments/eval_generative_mmlu_vllm.py --model-id "$dir" --n-per-subject 100 \
    --out "results/gen_mmlu_${TAG}_${cond}.json"
done

echo "[eval] DONE. Safety=full AdvBench520, ARC=full 25-shot, generative MMLU=400Q."
echo "       Judge *_gen/generations.jsonl locally; gen_mmlu_*.json has parse_rate."
