#!/usr/bin/env bash
# Back up version_A checkpoints to the PRIVATE HF repo (dual-use weights, per CLAUDE.md).
# Not using scripts/push_to_hf.py -- CLAUDE.md marks it stale.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/hf_backup.log
echo "=== HF backup: $(date -u) ===" | tee "$LOG"
python -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub

python - <<'PY' 2>&1 | tee -a "$LOG"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "aaronrockmenezes/tamperforge"
base = "version_a_2026_07_31"
files = {
    "outputs/version_a_qwen_500.pt.s400.pt": f"{base}/version_a_qwen_s400.pt",
    "outputs/version_a_qwen_500.pt.s450.pt": f"{base}/version_a_qwen_s450.pt",
    "outputs/version_a_qwen_500.pt":         f"{base}/version_a_qwen_s500.pt",
}
info = api.repo_info(repo, repo_type="model")
assert info.private, "repo is NOT private -- refusing to upload dual-use weights"
print(f"repo {repo} private={info.private}")
for src, dst in files.items():
    if not os.path.exists(src):
        print(f"!! missing {src}"); continue
    print(f"uploading {src} -> {dst} ({os.path.getsize(src)/1e9:.2f} GB)", flush=True)
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="model")
    print(f"  done {dst}", flush=True)
PY
echo "=== DONE $(date -u) ===" | tee -a "$LOG"
