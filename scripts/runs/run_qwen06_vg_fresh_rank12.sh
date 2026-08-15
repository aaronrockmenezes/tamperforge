#!/usr/bin/env bash
# Qwen3-0.6B Version G: fresh all-layer rank-1/rank-2 search, then AdvBench-520.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
PY="${PY:-python}"
[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python

MODEL="Qwen/Qwen3-0.6B"
CKPT="outputs/version_g_qwen_500.pt"
HF_FILE="final_backup_2026_08_03/checkpoints/version_g_qwen_500.pt"
RESTORE="outputs/hf_restore_qwen06_vg"
CLEAN="outputs/qwen06_vg_hf_clean_fresh"
mkdir -p outputs "$RESTORE"

if [ ! -s "$CKPT" ]; then
  hf download aaronrockmenezes/tamperforge "$HF_FILE" --local-dir "$RESTORE"
  cp "$RESTORE/$HF_FILE" "$CKPT"
fi
[ -s "$CKPT" ] || { echo "Qwen Version G checkpoint download failed" >&2; exit 1; }

if [ ! -s "$CLEAN/model.safetensors" ]; then
  "$PY" -u experiments/save_p1b_checkpoint.py \
    --model-id "$MODEL" --checkpoint "$CKPT" --attack none --out "$CLEAN"
fi
[ -s "$CLEAN/model.safetensors" ] || { echo "Qwen clean materialization failed" >&2; exit 1; }

MODEL_DIR="$CLEAN" SHORT=qwen06_vg_fresh_rank12 GPU=0 UTIL=0.30 \
PORTS=8765,8766,8767 ATTACK_RANKS=1,2 MATRIX_VARIANTS=rank1,rank2 \
RANK_K_ESTIMATOR=arditi_residual SWEEP_N=16 SWEEP_MAX_NEW=1024 \
SWEEP_SNAPSHOT_DEVICE=model GEN_WORKERS=64 JUDGE_WORKERS=64 \
JUDGE_MAX_TOKENS=1024 JUDGE_TIMEOUT_SECONDS=10 SAFETY_MAX_NEW_TOKENS=1024 \
SERVER_START_STAGGER_SECONDS=50 \
exec bash scripts/runs/eval_version_g_rank_matrix_3way.sh
