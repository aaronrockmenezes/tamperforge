#!/usr/bin/env bash
# Fetch the retained clean Phi-4 Mini Version G model, then run the 10-arm rank matrix.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
PY="${PY:-python}"
[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python

BUNDLE="outputs/hf_phi4mini_version_g"
MODEL_DIR="$BUNDLE/version_g_final/phi4mini/clean"
if [ ! -s "$MODEL_DIR/model.safetensors" ]; then
  "$PY" - "$BUNDLE" <<'PY'
import os, sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="aaronrockmenezes/tamperforge",
    repo_type="model",
    token=os.environ.get("HF_TOKEN"),
    local_dir=sys.argv[1],
    allow_patterns=["version_g_final/phi4mini/clean/*"],
)
PY
fi
[ -s "$MODEL_DIR/model.safetensors" ] || { echo "Phi clean download incomplete" >&2; exit 1; }

MODEL_DIR="$MODEL_DIR" SHORT="phi4mini_vg" GPU=0 UTIL=0.30 \
PORTS=8765,8766,8767 ATTACK_RANKS=1,2,4,8,16 \
RANK_K_ESTIMATOR=arditi_residual GEN_WORKERS=64 JUDGE_WORKERS=64 \
JUDGE_MAX_TOKENS=1024 JUDGE_TIMEOUT_SECONDS=10 \
SAFETY_MAX_NEW_TOKENS=1024 \
exec bash scripts/runs/eval_version_g_rank_matrix_3way.sh
