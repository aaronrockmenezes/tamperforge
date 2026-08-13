#!/usr/bin/env bash
set -euo pipefail

cd /workspace/tamperforge
set -a
. ./.env
set +a
exec /venv/main/bin/python -u scripts/runs/judge_version_g_extended_async.py
