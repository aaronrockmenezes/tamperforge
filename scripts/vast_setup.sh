#!/usr/bin/env bash
# Vast.ai GPU setup for tamperforge.
#
# Run from the tamperforge repo root:
#   bash scripts/vast_setup.sh
#
# Assumes a recent PyTorch CUDA image. For RTX 5090/Blackwell, prefer images
# with CUDA 12.8+ and recent PyTorch. If Torch CUDA is missing, set:
#   INSTALL_TORCH=1 bash scripts/vast_setup.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
ALLOW_NO_CUDA="${ALLOW_NO_CUDA:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

if [ ! -f "pyproject.toml" ] || [ ! -d "src/tamperforge" ]; then
  echo "ERROR: run this from the tamperforge repo root." >&2
  exit 1
fi

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

echo "==> [1/8] System/GPU"
uname -a || true
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. This is not a usable Vast GPU image yet." >&2
fi

echo "==> [2/8] Python"
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" -m pip --version
if [ "${SKIP_INSTALL}" = "1" ]; then
  echo "    SKIP_INSTALL=1; not modifying Python packages"
else
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
fi

echo "==> [3/8] Torch CUDA check"
if "${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY
then
  echo "    Torch CUDA OK"
else
  if [ "${INSTALL_TORCH:-0}" = "1" ]; then
    echo "    Installing CUDA Torch from ${TORCH_INDEX_URL}"
    "${PYTHON_BIN}" -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"
    "${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False after install"
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY
  elif [ "${ALLOW_NO_CUDA}" = "1" ]; then
    echo "    ALLOW_NO_CUDA=1; continuing without CUDA for dry-run validation"
  else
    echo "ERROR: Torch CUDA unavailable. Use a PyTorch CUDA 12.8+ image or rerun with INSTALL_TORCH=1." >&2
    exit 1
  fi
fi

echo "==> [4/8] Install tamperforge + eval deps"
if [ "${SKIP_INSTALL}" = "1" ]; then
  echo "    SKIP_INSTALL=1; skipping package install"
else
  "${PYTHON_BIN}" -m pip install -e ".[dev,eval]"
  "${PYTHON_BIN}" -m pip install --upgrade "lm_eval[hf]" accelerate sentencepiece
fi

echo "==> [5/8] CLI checks"
"${PYTHON_BIN}" - <<'PY'
import datasets, transformers, torch, tamperforge
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("tamperforge exports", len(tamperforge.__all__))
print("device", tamperforge.pick_device())
PY
if [ "${ALLOW_NO_CUDA}" != "1" ]; then
  "${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available()
PY
fi
if command -v lm_eval >/dev/null 2>&1; then
  lm_eval --help >/dev/null
elif [ "${SKIP_INSTALL}" = "1" ]; then
  echo "    SKIP_INSTALL=1; lm_eval CLI not required for dry-run"
else
  echo "ERROR: lm_eval CLI not found after install." >&2
  exit 1
fi

echo "==> [6/8] Hugging Face auth"
if [ -n "${HF_TOKEN:-}" ]; then
  hf auth login --token "${HF_TOKEN}" --add-to-git-credential
fi
if hf auth whoami >/dev/null 2>&1; then
  hf auth whoami
else
  echo "WARNING: HF auth missing. Run: hf auth login" >&2
fi

echo "==> [7/8] OpenRouter env"
if [ -n "${OPENROUTER_API_KEY:-}" ] && [ ! -f ".env" ]; then
  umask 077
  printf "OPENROUTER_API_KEY=%s\n" "${OPENROUTER_API_KEY}" > .env
  echo "    wrote .env from OPENROUTER_API_KEY"
elif [ -f ".env" ]; then
  echo "    .env exists"
else
  echo "    no OPENROUTER_API_KEY yet; judge commands will fail until .env is set"
fi

echo "==> [8/8] Create artifact dirs + smoke imports"
mkdir -p results outputs
PYTHONPATH=src "${PYTHON_BIN}" experiments/p0_baseline_eval.py --help >/dev/null
PYTHONPATH=src "${PYTHON_BIN}" experiments/judge_generations.py --help >/dev/null
PYTHONPATH=src "${PYTHON_BIN}" experiments/p1_mad_crux.py --help >/dev/null

echo "==> Setup complete."
echo "Next: run the commands in docs/vast_runbook.md."
