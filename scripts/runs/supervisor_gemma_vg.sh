#!/usr/bin/env bash
set -euo pipefail
cd /workspace/tamperforge
exec bash scripts/runs/overnight_gemma_vg.sh
