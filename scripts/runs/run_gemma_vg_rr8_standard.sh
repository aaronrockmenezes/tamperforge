#!/usr/bin/env bash
# Standard-only evaluation for Gemma Version G lambda_rr=8.
# Uses unique vgg8_* artifact names; the frozen extended suite is intentionally excluded.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/eval logs/heretic logs/probes results outputs

LOG=logs/eval/gemma_vg_rr8_standard_$(date -u +%Y%m%dT%H%M%S).log
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

[ -s outputs/version_g_gemma_rr8_500.pt ] || {
  say "[FAIL] missing raw rr=8 checkpoint"
  exit 1
}

say "=== Gemma Version G rr=8 standard battery ==="
TAG=version_g_gemma_rr8_500 SHORT=vgg8 WAIT_ON=none \
  MODEL_ID=google/gemma-3-1b-it DIRECTION_LAYER=14 \
  BASE_TAG=gbase_clean BASE_HF=google/gemma-3-1b-it \
  ENFORCE_GATES=0 JUDGE_WORKERS=96 SKIP_MMLU=1 \
  bash scripts/runs/chain_f.sh 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
say "chain_rc=$rc"
[ "$rc" = 0 ] || exit "$rc"

say "=== rr=8 standard battery complete ==="
