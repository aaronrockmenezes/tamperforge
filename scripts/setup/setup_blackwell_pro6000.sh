#!/usr/bin/env bash
# Setup/verify a Vast RTX PRO 6000 Blackwell box for TamperForge Qwen3-8B.
#
# Run from the tamperforge repo root after cloning/syncing the repo:
#   bash scripts/setup_blackwell_pro6000.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x /venv/main/bin/python ]; then
  export PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
else
  export PYTHON_BIN="${PYTHON_BIN:-python}"
fi

echo "==> Blackwell PRO 6000 verification"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "ERROR: nvidia-smi not found; this is not a usable GPU image." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import re
import subprocess
import sys

smi = subprocess.check_output(["nvidia-smi", "-L"], text=True)
print(smi.strip())
if not re.search(r"RTX\s+PRO\s+6000|RTX\s+6000", smi, re.I):
    print("WARNING: did not see RTX PRO 6000 in nvidia-smi -L", file=sys.stderr)
PY

echo "==> Base TamperForge setup"
PYTHON_BIN="$PYTHON_BIN" \
  SKIP_VLLM_INSTALL="${SKIP_VLLM_INSTALL:-1}" \
  bash scripts/vast_setup.sh

echo "==> Qwen3-8B thinking-mode smoke"
PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
text = tok.apply_chat_template(
    [{"role": "user", "content": "Say ready."}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
print(text[-300:])
assert "<think>" not in text or "<|im_start|>assistant" in text
print("Qwen3 chat template accepts enable_thinking=True")
PY

echo "==> vLLM Qwen3-8B load smoke"
PYTHONPATH=src "$PYTHON_BIN" experiments/p0_baseline_eval.py \
  --backend vllm \
  --model-id Qwen/Qwen3-8B \
  --prompt-source advbench \
  --advbench-source walledai \
  --n-prompts 2 \
  --max-new-tokens 128 \
  --max-length 4096 \
  --vllm-batch-size 2 \
  --vllm-gpu-memory-utilization 0.82 \
  --qwen-thinking on \
  --vllm-temperature 0.6 \
  --vllm-top-p 0.95 \
  --vllm-top-k 20 \
  --n-arc 0 \
  --run-id smoke_qwen3_8b_thinking_vllm

echo "==> Setup complete. Next: bash scripts/qwen3_8b_thinking_dl_sweep.sh"
