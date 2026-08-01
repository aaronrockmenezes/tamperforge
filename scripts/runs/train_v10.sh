#!/usr/bin/env bash
# ABL-v10 micro-run: tamper-triggered fail-closed output.
#
# Qwen3-0.6B first probe (exact v8 attack distribution):
#   MODEL=Qwen/Qwen3-0.6B DL=20 \
#   OUT=outputs/qwen3_0p6b_v10_v8control.pt \
#   PROFILE=v8 bash scripts/train_v10.sh
#
# PROFILE: v8 | partial_shared | perlayer_full | partial_perlayer | mixed
# Run axes separately before mixed. Default STEPS=150 is diagnostic, not a release run.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:?set MODEL}"
DL="${DL:?set DL (judged refusal-layer peak)}"
OUT="${OUT:?set OUT}"
PROFILE="${PROFILE:-v8}"

ATTACK_ARGS=(--attack-profile "$PROFILE")
if [[ "$PROFILE" != "v8" ]]; then
  ATTACK_LAYERS="${ATTACK_LAYERS:?set ATTACK_LAYERS to a base-validated coherent-harm band}"
  ATTACK_ARGS+=(--attack-layers "$ATTACK_LAYERS")
fi

"${PY:-python}" experiments/train_tamper_resistant_v8.py \
  --model-id "$MODEL" \
  --out "$OUT" \
  --train-scope all \
  --abliterate-layers all \
  --attack-ensemble \
  "${ATTACK_ARGS[@]}" \
  --attack-alpha-min "${ALPHA_MIN:-0.2}" \
  --attack-alpha-max "${ALPHA_MAX:-0.6}" \
  --direction-layer "$DL" \
  --recompute-direction-every 25 \
  --lambda-shutdown "${LSHUTDOWN:-4}" \
  --shutdown-target "${SHUTDOWN_TARGET:-}" \
  --shutdown-benign-prompts "${SHUTDOWN_BENIGN:-2}" \
  --shutdown-harmful-prompts "${SHUTDOWN_HARMFUL:-2}" \
  --lambda-gib 0 \
  --stage2-lambda-gib 0 \
  --lambda-uncensor 0 \
  --lambda-harm 0 \
  --lambda-rr 0 \
  --lambda-safe "${LSAFE:-4}" \
  --lambda-reg "${LREG:-0.1}" \
  --lambda-clean "${LCLEAN:-3}" \
  --clean-gen-prompts 2 \
  --clean-gen-tokens 32 \
  --clean-start-step 0 \
  --clean-ramp-steps 0 \
  --ifeval-in-loop \
  --ifeval-probe-n 24 \
  --save-every 25 \
  --steps "${STEPS:-150}" \
  --eval-every 25 \
  --lr "${LR:-1e-5}" \
  --seed "${SEED:-42}"

echo "### v10 diagnostic complete: $OUT"
echo "### Do not promote without fixed rank-1 + replayed Heretic + fresh Heretic evaluation."
