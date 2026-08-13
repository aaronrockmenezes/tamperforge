#!/usr/bin/env bash
# Finish the untouched Gemma base clean standard battery, then resume rr=8.
# MMLU is intentionally excluded from both arms by user request.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/eval results

LOG=logs/eval/gemma_base_then_rr8_standard_$(date -u +%Y%m%dT%H%M%S).log
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for the already-started base GSM8K control"
while supervisorctl status tamperforge-gemma-base-gsm8k 2>/dev/null | grep -q RUNNING; do
  sleep 20
done

say "=== untouched Gemma base clean standard battery (MMLU excluded) ==="
SERVE_LL=api SKIP_MMLU=1 JUDGE_WORKERS=96 \
  bash scripts/eval/serve_eval.sh gbase_clean google/gemma-3-1b-it off 2>&1 | tee -a "$LOG"
base_rc=${PIPESTATUS[0]}
say "base_standard_rc=$base_rc"
[ "$base_rc" = 0 ] || exit "$base_rc"

say "=== resume Gemma rr=8 standard battery (MMLU excluded) ==="
SKIP_MMLU=1 bash scripts/runs/run_gemma_vg_rr8_standard.sh 2>&1 | tee -a "$LOG"
rr8_rc=${PIPESTATUS[0]}
say "rr8_standard_rc=$rr8_rc"
[ "$rr8_rc" = 0 ] || exit "$rr8_rc"

say "=== base then rr=8 standard batteries complete ==="
