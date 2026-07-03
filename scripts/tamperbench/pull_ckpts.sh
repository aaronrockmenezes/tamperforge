#!/usr/bin/env bash
# Pull the 3 forged ABL-v7 ckpts from the private HF repo onto a fresh box.
# gemma v7 is usually already present (it ships with the OG runs); qwen/llama
# are the multimodel ckpts. Idempotent — skips files already on disk.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="${REPO:-aaronrockmenezes/tamperforge}"
mkdir -p outputs
for f in \
  tamper_resistant_p1b_v7.pt \
  tamper_resistant_qwen3_0p6b_v7.pt \
  tamper_resistant_llama32_1b_v7_L13.pt ; do
  if [[ -f "outputs/$f" ]]; then echo "have  outputs/$f"; continue; fi
  echo "pull  $REPO:$f"
  hf download "$REPO" "outputs/$f" --repo-type model --local-dir . || \
    echo "!! missing $f on HF — check scripts/push_to_hf.py ran for it"
done
ls -la outputs/tamper_resistant_*_v7*.pt outputs/tamper_resistant_p1b_v7.pt 2>/dev/null || true
