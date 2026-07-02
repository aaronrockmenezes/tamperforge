#!/usr/bin/env bash
# Fresh 2x RTX 5090 (Blackwell, CUDA 13+) setup for tamperforge.
# Run from the repo root AFTER cloning. Export secrets first:
#   export HF_TOKEN=...             # gated gemma + private tamperforge HF repo
#   export OPENROUTER_API_KEY=...   # judge gate + eval judging
#   export GH_PAT=...               # (optional) if you want box git push
#   bash scripts/setup_5090.sh
set -uo pipefail
PY="${PY:-/venv/main/bin/python}"
command -v "$PY" >/dev/null 2>&1 || PY=python
PIP="$PY -m pip"
echo "[setup] python = $PY"

# 0) sanity: torch + CUDA + how many GPUs
$PY - <<'PY' || { echo "[setup] torch/CUDA missing -> use a PyTorch-CUDA-13 image, or run scripts/vast_setup.sh INSTALL_TORCH=1"; }
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
print("gpus", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

# 1) deps (torch/vllm assumed from the image; add the rest we use)
echo "[setup] installing deps..."
$PIP install -q -U peft datasets "lm_eval>=0.4.9" transformers huggingface_hub tqdm requests || echo "[setup] pip warnings ^"

# 2) HF auth (gated gemma + private ckpt repo)
if [ -n "${HF_TOKEN:-}" ]; then
  $PY -c "from huggingface_hub import login; login('${HF_TOKEN}')" && echo "[setup] HF logged in"
else
  echo "[setup] WARN: HF_TOKEN not set -> gated gemma + private ckpt pull will fail"
fi

# 3) judge key -> .env
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
  grep -q OPENROUTER_API_KEY .env 2>/dev/null || echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}" >> .env
  echo "[setup] OPENROUTER_API_KEY -> .env"
else
  echo "[setup] WARN: OPENROUTER_API_KEY not set -> judge gate/eval will 401"
fi

# 4) git identity (for optional box push)
git config user.email "85219711+aaronrockmenezes@users.noreply.github.com" 2>/dev/null
git config user.name "aaronrockmenezes" 2>/dev/null
[ -n "${GH_PAT:-}" ] && git remote set-url origin "https://${GH_PAT}@github.com/aaronrockmenezes/tamperforge.git"

# 5) pull ABL-v7 checkpoint from private HF (not in git — too big)
echo "[setup] pulling ABL-v7 ckpt from HF..."
mkdir -p outputs
$PY - <<'PY'
import shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download("aaronrockmenezes/tamperforge", "adapters/tamper_resistant_p1b_v7.pt", repo_type="model")
shutil.copy(p, "outputs/tamper_resistant_p1b_v7.pt")
print("[setup] outputs/tamper_resistant_p1b_v7.pt ready")
PY

# 6) the demos file is committed in git (results/p1b_v7_base_att_gen/generations.jsonl)
test -f results/p1b_v7_base_att_gen/generations.jsonl \
  && echo "[setup] demos present ($(wc -l < results/p1b_v7_base_att_gen/generations.jsonl) rows)" \
  || echo "[setup] WARN: demos file missing (git pull?)"

echo "[setup] DONE. Smoke: see the 1-step smoke in docs/setup_5090.md"
