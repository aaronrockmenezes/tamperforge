#!/usr/bin/env bash
set -euo pipefail

cd /workspace/tamperforge
exec bash scripts/runs/run_version_g_extended_heldout.sh
