#!/usr/bin/env bash
# Finish the already-generated plain rank-1/rank-2 cells first, then resume the full matrix.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODEL_DIR="outputs/hf_phi4mini_version_g/version_g_final/phi4mini/clean"
[ -s "$MODEL_DIR/model.safetensors" ] || { echo "missing Phi clean model" >&2; exit 1; }

MODEL_DIR="$MODEL_DIR" SHORT=phi4mini_vg GPU=0 UTIL=0.30 \
PORTS=8765,8766,8767 ATTACK_RANKS=1,2 MATRIX_VARIANTS=rank1,rank2 \
SWEEP_DIR=results/dl_sweeps/phi4mini_vg_rank12_early \
SWEEP_GEN_DIR=results/dl_sweeps/phi4mini_vg_rank_matrix/generations \
RANK_K_ESTIMATOR=arditi_residual GEN_WORKERS=64 JUDGE_WORKERS=64 \
JUDGE_MAX_TOKENS=1024 JUDGE_TIMEOUT_SECONDS=10 SAFETY_MAX_NEW_TOKENS=1024 \
bash scripts/runs/eval_version_g_rank_matrix_3way.sh

exec bash scripts/runs/run_phi4mini_rank_matrix.sh
