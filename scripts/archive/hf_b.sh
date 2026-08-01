#!/usr/bin/env bash
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/hf_backup_b.log
echo "=== HF backup version_B $(date -u) ===" | tee "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "aaronrockmenezes/tamperforge"
base = "version_b_2026_08_01"
files = {
    "outputs/version_b_qwen_500.pt.s400.pt": f"{base}/version_b_qwen_s400.pt",
    "outputs/version_b_qwen_500.pt.s450.pt": f"{base}/version_b_qwen_s450.pt",
    "outputs/version_b_qwen_500.pt":         f"{base}/version_b_qwen_s500.pt",
}
info = api.repo_info(repo, repo_type="model")
assert info.private, "repo is NOT private -- refusing to upload dual-use weights"
for src, dst in files.items():
    if not os.path.exists(src): print(f"!! missing {src}"); continue
    print(f"uploading {src} ({os.path.getsize(src)/1e9:.2f} GB)", flush=True)
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="model")
    print(f"  done {dst}", flush=True)
PY
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
