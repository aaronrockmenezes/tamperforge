#!/usr/bin/env bash
# Move completed, non-Gemma logs into a model/campaign/phase archive.
#
# Safe defaults:
#   MODE=dry-run (default) prints the proposed mapping.
#   MODE=apply moves files and records every move in logs/archive/MANIFEST.tsv.
#   MODE=restore replays the manifest backwards.
#
# Gemma/vgg/gbase/gext logs and files held open by live processes are never moved.
set -euo pipefail
cd /workspace/tamperforge

MODE="${MODE:-dry-run}"
ARCHIVE=logs/archive
MANIFEST="$ARCHIVE/MANIFEST.tsv"

case "$MODE" in
  dry-run|apply|restore) ;;
  *) echo "MODE must be dry-run, apply, or restore" >&2; exit 2 ;;
esac

mkdir -p "$ARCHIVE"

if [ "$MODE" = restore ]; then
  [ -s "$MANIFEST" ] || { echo "No manifest at $MANIFEST" >&2; exit 1; }
  tac "$MANIFEST" | while IFS=$'\t' read -r src dst; do
    [ "$src" = source ] && continue
    [ -e "$dst" ] || continue
    mkdir -p "$(dirname "$src")"
    [ ! -e "$src" ] || { echo "SKIP restore collision: $src"; continue; }
    mv "$dst" "$src"
    echo "restored  $dst -> $src"
  done
  exit 0
fi

OPEN_LOGS=$(mktemp)
trap 'rm -f "$OPEN_LOGS"' EXIT
find /proc/[0-9]*/fd -type l -lname '/workspace/tamperforge/logs/*' \
  -printf '%l\n' 2>/dev/null | sort -u > "$OPEN_LOGS" || true

model_for () {
  local n="$1"
  case "$n" in
    *llama*|*vgl*|*chain_gl*|*arl*|*shl*|*lbase*|*lceil*|*hlvb*|*hvc*llama*|*lvbe*) echo llama ;;
    *qwen*|*version_a*|*version_b*|*version_c*|*version_d*|*version_e*|*version_f*|*version_g*|*chain_g_*|*arq*|*shq*|*xbase*|*xva*|*xvb*|*xvc*|*vgho*|*serve_vg*|*serve_vf*|*serve_ve*|*vg_*|*vf_*|*ve_*|*vb_*|*vc_*) echo qwen ;;
    *) echo shared ;;
  esac
}

campaign_for () {
  local n="$1"
  case "$n" in
    *benign_ft*|*harmful_ft*|*harm_ft*|*sft*|*lora*|*vgho*) echo finetuning_attacks ;;
    *version_g*|*serve_vg*|*chain_g*|*smoke5_vg*|*heretic_vg*) echo version_g ;;
    *version_f*|*serve_vf*|*chain_f*) echo version_f ;;
    *version_e*|*serve_ve*|*overnight_e*|*heretic_ve*) echo version_e ;;
    *version_d*) echo version_d ;;
    *version_c*|*vc_*|*hvc*|*xvc*|*reeval_vc*) echo version_c ;;
    *version_b*|*vb_*|*hvb*|*xvb*|*eval_vb*) echo version_b ;;
    *version_a*|*va_*|*xva*|*vavc*) echo version_a ;;
    *art*|*arq*|*arl*) echo art ;;
    *shairah*|*shq*|*shl*) echo shairah ;;
    *replicate*|*rep_*) echo replication ;;
    *extended*|*heldout*) echo extended ;;
    *base*|*ceiling*|*ceil*) echo baselines ;;
    *panel*) echo panels_legacy ;;
    *backup*|*upload*|*status*|*freeze*) echo operations ;;
    *) echo misc ;;
  esac
}

phase_for () {
  local p="$1"
  case "$p" in
    logs/training_runs/*) echo training ;;
    logs/eval/vllm/*) echo evaluation/vllm ;;
    logs/eval/*) echo evaluation/harness ;;
    logs/heretic/*) echo attacks/heretic ;;
    logs/probes/*) echo diagnostics/probes ;;
    logs/drivers/*) echo operations/drivers ;;
    logs/ops/*) echo operations/storage ;;
    logs/panels/*) echo diagnostics/panels_legacy ;;
    logs/tamperbench/*) echo evaluation/tamperbench ;;
    *) echo misc ;;
  esac
}

if [ "$MODE" = apply ] && [ ! -e "$MANIFEST" ]; then
  printf 'source\tdestination\n' > "$MANIFEST"
fi

moved=0
skipped_gemma=0
skipped_open=0
while IFS= read -r src; do
  name=$(basename "$src")
  lower=$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]')

  # Protect all current and historical Gemma logs. The current rr=8 chain also uses
  # vgg8 names that do not contain the word "gemma".
  case "$lower" in
    *gemma*|*vgg*|*gbase*|*gext*|*gg_clean*|*gg_her*)
      skipped_gemma=$((skipped_gemma + 1))
      continue
      ;;
  esac

  absolute="/workspace/tamperforge/$src"
  if grep -Fxq "$absolute" "$OPEN_LOGS"; then
    echo "SKIP open  $src"
    skipped_open=$((skipped_open + 1))
    continue
  fi

  model=$(model_for "$lower")
  campaign=$(campaign_for "$lower")
  phase=$(phase_for "$src")
  dst="$ARCHIVE/$model/$campaign/$phase/$name"

  [ "$src" != "$dst" ] || continue
  if [ -e "$dst" ]; then
    echo "SKIP collision  $src -> $dst" >&2
    continue
  fi

  if [ "$MODE" = dry-run ]; then
    echo "$src -> $dst"
  else
    mkdir -p "$(dirname "$dst")"
    mv "$src" "$dst"
    printf '%s\t%s\n' "$src" "$dst" >> "$MANIFEST"
  fi
  moved=$((moved + 1))
done < <(find logs/training_runs logs/eval logs/heretic logs/probes logs/drivers \
              logs/ops logs/panels logs/tamperbench \
              -type f ! -path 'logs/archive/*' -print | sort)

echo "mode=$MODE selected=$moved protected_gemma=$skipped_gemma protected_open=$skipped_open"
[ "$MODE" != apply ] || echo "manifest=$MANIFEST"
