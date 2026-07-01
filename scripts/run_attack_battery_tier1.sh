#!/usr/bin/env bash
# Tier-1 abliteration-attack battery against the P1b-A v6 checkpoint.
# Saves each attacked variant as an HF dir (pure weights) and generates safety
# responses via vLLM (batched, fast). Judge the generations LOCALLY afterward.
#
# Run on the GPU box:  bash scripts/run_attack_battery_tier1.sh
# Self-updates code first via checkout (avoids the untracked-results pull clash).

set -euo pipefail
cd /workspace/tamperforge

# grab latest CODE only (not results) -> no merge clash with untracked result dirs
git fetch origin -q
git checkout origin/main -- experiments src scripts docs
echo "[battery] code updated to origin/main"

CKPT=outputs/tamper_resistant_p1b_v6.pt

gen() {  # $1 = model dir, $2 = run-id  -- FULL AdvBench (all 520)
  python experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
    --advbench-source walledai --n-advbench 520 --n-arc 0 \
    --max-new-tokens 128 --max-length 4096 --run-id "$2"
}
arc() {  # $1 = model dir, $2 = tag  -- FULL ARC-Challenge 25-shot test (1172, no --limit)
  lm_eval --model vllm \
    --model_args pretrained="$1",dtype=bfloat16,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
    --tasks arc_challenge --num_fewshot 25 --batch_size 1 \
    --output_path "results/lm_eval_${2}_arc25_full" --log_samples
}
save() { python experiments/save_p1b_checkpoint.py "$@"; }

run_variant() {  # $1 = out dir, $2 = tag, then save args...
  local dir="$1" tag="$2"; shift 2
  save "$@" --out "$dir"
  gen "$dir" "${tag}_gen"
  arc "$dir" "$tag"
}

# ---- 1.2 layer-subset (attacker skips layers) ----
run_variant outputs/p1b_v6_att_L13-25 p1b_v6_att_L13-25 --checkpoint $CKPT --attack all --abliterate-layers 13-25
run_variant outputs/p1b_v6_att_L0-12  p1b_v6_att_L0-12  --checkpoint $CKPT --attack all --abliterate-layers 0-12

# ---- 1.3 rank-k subspace (SVD) ----
run_variant outputs/p1b_v6_att_svd4 p1b_v6_att_svd4 --checkpoint $CKPT --attack all --attack-estimator svd --attack-rank 4
run_variant outputs/p1b_v6_att_svd8 p1b_v6_att_svd8 --checkpoint $CKPT --attack all --attack-estimator svd --attack-rank 8

# ---- 1.4 whitened-SVD + different prompt sample ----
run_variant outputs/p1b_v6_att_wsvd4 p1b_v6_att_wsvd4 --checkpoint $CKPT --attack all --attack-estimator whitened_svd --attack-rank 4
run_variant outputs/p1b_v6_att_seed7 p1b_v6_att_seed7 --checkpoint $CKPT --attack all --attack-estimator diffmeans --direction-seed 7

# ---- base controls (undefended) for the stronger estimators ----
run_variant outputs/p1b_v6_base_svd8  p1b_v6_base_svd8  --attack all --attack-estimator svd --attack-rank 8
run_variant outputs/p1b_v6_base_wsvd4 p1b_v6_base_wsvd4 --attack all --attack-estimator whitened_svd --attack-rank 4

echo "[battery] DONE. Full AdvBench 520 gens + full ARC-25 per variant. Judge locally."
