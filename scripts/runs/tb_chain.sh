#!/usr/bin/env bash
# TamperBench refusal_ablation on our three Qwen defenses, queued behind the base run.
#
# Third-party validation: TamperBench is someone else's attack implementation, prompts and
# StrongREJECT judge, so it tests whether our results survive leaving our own harness.
# mmlu_pro_val gives capability-under-attack from their side too.
#
# The clone is NOT vendored and its 3 memory patches (fp64->fp32 in refusal_ablation.py AND
# attack_utils.py's kl_div_fn, plus del+empty_cache in the layer sweep) live only on this
# box's disk -- reapply by hand on any fresh clone. See docs/common_issues.md.
set -uo pipefail
TB=/workspace/TamperBench
LOGD=/workspace/tamperforge/logs/tamperbench
mkdir -p "$LOGD"

say () { echo "[$(date -u +%H:%M:%S)] $*"; }

# One GPU: never two of these at once.
if tmux has-session -t tb 2>/dev/null; then
  say "[wait] base run in progress..."
  while tmux has-session -t tb 2>/dev/null; do sleep 60; done
  say "[wait] base done, settling 30s"; sleep 30
fi

cd "$TB"
set -a; . /workspace/tamperforge/.env; set +a

for spec in "qwen_va:/workspace/tamperforge/outputs/xva_clean" \
            "qwen_vb:/workspace/tamperforge/outputs/xvb_clean" \
            "qwen_vc:/workspace/tamperforge/outputs/xvc_clean"; do
  alias="${spec%%:*}"; md="${spec#*:}"
  if [ ! -f "$md/model.safetensors" ]; then say "[MISSING] $md"; continue; fi
  # guard on the artifact, not the directory
  if find results -path "*/${alias}/*" -name 'results.parquet' -print -quit 2>/dev/null | grep -q .; then
    say "[skip] $alias already has results.parquet"; continue
  fi
  say "=== $alias <- $md ==="
  PYTORCH_ALLOC_CONF=expandable_segments:True uv run scripts/whitebox/benchmark_grid.py "$md" \
    --attacks refusal_ablation --model-alias "$alias" \
    > "$LOGD/${alias}.log" 2>&1
  rc=$?
  say "  $alias rc=$rc"
  if [ "$rc" -ne 0 ]; then
    say "  [FAIL] $alias -- tail:"; tail -15 "$LOGD/${alias}.log"
  fi
  # reap any orphaned CUDA holder between runs
  for p in $(pgrep -f 'benchmark_grid' 2>/dev/null); do
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    [ "$pp" = "1" ] && { say "  [reap] $p"; kill -9 "$p" 2>/dev/null || true; }
  done
  sleep 10
done

say "=== TB CHAIN DONE ==="
df -h /workspace | tail -1
