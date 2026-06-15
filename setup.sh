#!/usr/bin/env bash
# tamperforge environment setup + smoke test.
# Uses the existing env_ml conda env — does NOT create a new venv/uv.
set -euo pipefail

ENV_NAME="env_ml"
PY="/Users/aaronrockmenezes/miniforge3/envs/${ENV_NAME}/bin/python"

echo "==> [1/4] Checking conda env '${ENV_NAME}'"
if [ ! -x "$PY" ]; then
  echo "ERROR: ${PY} not found. Activate or create env_ml first (see AGENTS rule: never scaffold a new venv)." >&2
  exit 1
fi
echo "    python: $($PY --version)"

echo "==> [2/4] Installing tamperforge (editable) + deps"
"$PY" -m pip install -e ".[dev]" --quiet

echo "==> [3/4] Checking Hugging Face auth (Gemma 3 is gated)"
if ! "$PY" -c "from huggingface_hub import whoami; whoami()" 2>/dev/null; then
  echo "    WARNING: not logged in. Run: hf auth login" >&2
else
  echo "    HF auth OK"
fi

echo "==> [4/4] Smoke test: import + device + SAE metadata"
"$PY" - <<'PY'
import tamperforge as tf
dev = tf.pick_device()
print(f"    device: {dev}")
print(f"    import OK — exports: {len(tf.__all__)} symbols")
PY

echo "==> Setup complete. Next: python experiments/p1_mad_crux.py --help"
