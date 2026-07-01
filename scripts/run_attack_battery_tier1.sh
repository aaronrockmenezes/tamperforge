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

gen() {  # $1 = model dir, $2 = run-id
  python experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
    --advbench-source walledai --n-advbench 200 --n-arc 0 \
    --max-new-tokens 128 --max-length 4096 --run-id "$2"
}
save() {  # remaining args -> save_p1b_checkpoint.py
  python experiments/save_p1b_checkpoint.py "$@"
}

# ---- 1.2 layer-subset (attacker skips layers) ----
save --checkpoint $CKPT --attack all --abliterate-layers 13-25 --out outputs/p1b_v6_att_L13-25
gen  outputs/p1b_v6_att_L13-25 p1b_v6_att_L13-25_gen
save --checkpoint $CKPT --attack all --abliterate-layers 0-12  --out outputs/p1b_v6_att_L0-12
gen  outputs/p1b_v6_att_L0-12  p1b_v6_att_L0-12_gen

# ---- 1.3 rank-k subspace (SVD) ----
save --checkpoint $CKPT --attack all --attack-estimator svd --attack-rank 4 --out outputs/p1b_v6_att_svd4
gen  outputs/p1b_v6_att_svd4 p1b_v6_att_svd4_gen
save --checkpoint $CKPT --attack all --attack-estimator svd --attack-rank 8 --out outputs/p1b_v6_att_svd8
gen  outputs/p1b_v6_att_svd8 p1b_v6_att_svd8_gen

# ---- 1.4 whitened-SVD + different prompt sample ----
save --checkpoint $CKPT --attack all --attack-estimator whitened_svd --attack-rank 4 --out outputs/p1b_v6_att_wsvd4
gen  outputs/p1b_v6_att_wsvd4 p1b_v6_att_wsvd4_gen
save --checkpoint $CKPT --attack all --attack-estimator diffmeans --direction-seed 7 --out outputs/p1b_v6_att_seed7
gen  outputs/p1b_v6_att_seed7 p1b_v6_att_seed7_gen

# ---- base controls (undefended) for the stronger estimators ----
save --attack all --attack-estimator svd --attack-rank 8 --out outputs/p1b_v6_base_svd8
gen  outputs/p1b_v6_base_svd8 p1b_v6_base_svd8_gen
save --attack all --attack-estimator whitened_svd --attack-rank 4 --out outputs/p1b_v6_base_wsvd4
gen  outputs/p1b_v6_base_wsvd4 p1b_v6_base_wsvd4_gen

echo "[battery] DONE. Judge locally: scp each results/*_gen/generations.jsonl + judge_generations.py"
