#!/usr/bin/env bash
# Resumable end-to-end Gemma Version G campaign.
set -euo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/overnight_gemma_vg_$(date -u +%Y%m%dT%H%M%S).log
mkdir -p logs/training_runs logs/eval results outputs
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== STAGE 1 train Gemma Version G ==="
bash scripts/runs/run_version_g_gemma.sh 2>&1 | tee -a "$LOG"

say "=== STAGE 2 preserve raw checkpoint on private HF ==="
hf upload aaronrockmenezes/tamperforge outputs/version_g_gemma_500.pt \
  final_backup_2026_08_04/checkpoints/version_g_gemma_500.pt --repo-type model >>"$LOG" 2>&1

say "=== wait for pre-existing Qwen/Llama extended judge ==="
while supervisorctl status tamperforge-vgho-judge 2>/dev/null | grep -q RUNNING; do
  say "existing 96-worker judge still running"; sleep 60
done

say "=== STAGE 3 clean/rank-1/surgical/Heretic + standard battery ==="
STANDARD_DONE=results/.gemma_standard_generation_complete
if [ -e "$STANDARD_DONE" ]; then
  say "[skip] completed standard generations and capability results"
else
  TAG=version_g_gemma_500 SHORT=vgg WAIT_ON=none \
    MODEL_ID=google/gemma-3-1b-it DIRECTION_LAYER=14 \
    BASE_TAG=gbase_clean BASE_HF=google/gemma-3-1b-it \
    ENFORCE_GATES=0 JUDGE_WORKERS=96 \
    bash scripts/runs/chain_f.sh 2>&1 | tee -a "$LOG"
  for p in outputs/version_g_gemma_500_clean outputs/vgg_rank1 outputs/vgg_surg_k16 outputs/vgg_her_s0_att; do
    [ -s "$p/model.safetensors" ] || { say "[FAIL] missing $p"; exit 1; }
  done
  touch "$STANDARD_DONE"
fi

say "=== STAGE 4 preserve all Gemma Version G variants on private HF ==="
# Preserve the tiny Heretic replay recipe first. Together with the already-backed-up
# raw checkpoint it makes every materialised arm reproducible even if HF storage is
# temporarily full.
hf upload aaronrockmenezes/tamperforge results/vgg_her_s0_trial.json \
  final_backup_2026_08_04/version_g_gemma_variants/vgg_her_s0_trial.json \
  --repo-type model >>"$LOG" 2>&1

VARIANT_BACKUP_DEFER=results/.gemma_variant_backup_deferred
if [ -e "$VARIANT_BACKUP_DEFER" ]; then
  say "[defer] materialised variant backup previously hit the private-HF storage limit"
else
  # The clean materialisation is retained on HF. Heretic is reproducible from
  # the backed-up raw checkpoint plus vgg_her_s0_trial.json, so avoid storing a
  # second 2 GB materialisation.
  for p in outputs/version_g_gemma_500_clean; do
    if ! hf upload aaronrockmenezes/tamperforge "$p" \
      "final_backup_2026_08_04/version_g_gemma_variants/$(basename "$p")" \
      --repo-type model >>"$LOG" 2>&1; then
      say "[defer] private-HF variant upload failed; raw checkpoint + Heretic replay recipe are backed up"
      say "[defer] remove $VARIANT_BACKUP_DEFER after increasing HF storage, then rerun to resume uploads"
      touch "$VARIANT_BACKUP_DEFER"
      break
    fi
  done
fi

say "=== STAGE 5 frozen held-out extended suite + asynchronous judging ==="
python -u scripts/runs/judge_gemma_vg_extended_async.py >>logs/eval/gemma_vg_async_judge.log 2>&1 &
JPID=$!
bash scripts/runs/run_gemma_vg_extended.sh 2>&1 | tee -a "$LOG"
wait "$JPID"

say "=== FINAL VALIDATION ==="
python - <<'PY' | tee -a "$LOG"
from pathlib import Path
import json
r=Path("results")
std=["vgg_clean","vgg_rank1","vgg_surg_k16","vgg_her_s0"]
for t in std:
    p=r/f"{t}_judged"/"summary.json"
    print("standard", t, "OK" if p.exists() else "MISSING")
print("extended_judged", sum((r/f"gext_{t}_{b}_judged"/"summary.json").exists() for t in ["gbase_clean","gg_clean","gg_her"] for b in ["harmbench","jailbreakbench","strongreject","sorrybench","beavertails","xstest_unsafe"]), "/18")
print("extended_orscore", sum((r/f"gext_{t}_{b}_orscore"/"summary.json").exists() for t in ["gbase_clean","gg_clean","gg_her"] for b in ["xstest_safe","orbench"]), "/6")
PY
say "=== OVERNIGHT GEMMA CAMPAIGN COMPLETE ==="
