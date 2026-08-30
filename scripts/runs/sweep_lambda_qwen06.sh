#!/usr/bin/env bash
# Lambda sensitivity sweep, Qwen3-0.6B, new Version-G recipe.
#
# WHY: none of the seven active loss weights came from a sweep against an outcome.
# Provenance is "copy prior arm -> patch what broke -> freeze" (docs/handoff_2026_08_03_codex.md
# :226 and :303; LAMBDA_RR=4 is documented as "unswept" at :290). This produces the sensitivity
# table that a methods section needs.
#
# One factor at a time around the current recipe. Every arm is 600 steps: the wall transition
# on this model is concentrated in 400-650 (results/compiled/.../training_stop_analysis_20260827.md),
# so 600 is the cheapest step count that still separates "forms a wall" from "does not".
#
#   bash scripts/runs/sweep_lambda_qwen06.sh a0        # one arm
#   ARMS="a0 a1 a2" bash scripts/runs/sweep_lambda_qwen06.sh   # several, sequential
#
# GPU=1 to pin a second card and run two shells in parallel.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }

PY="${PY:-/venv/main/bin/python}"
MODEL_ID=Qwen/Qwen3-0.6B
STEPS="${STEPS:-600}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export TF_QWEN_THINKING=on
export PYTHONPATH=src

# ---- arm definitions: ONLY the deltas from the control -----------------------
# a0 is the current recipe verbatim; every other arm changes exactly one thing.
arm_flags() {
  case "$1" in
    a0) echo "" ;;                                   # control
    a1) echo "--lambda-harm 0" ;;                    # drop the dead term
    a2) echo "--harm-margin 12" ;;                   # make L_harm actually bind
    a3) echo "--lambda-rr 16" ;;                     # rr up   (doc: unswept, scale-mismatched)
    a4) echo "--lambda-rr 1" ;;                      # rr down
    a5) echo "--uncensor-margin 12" ;;               # make L_uncensor bind late
    *)  echo "unknown arm: $1" >&2; exit 2 ;;
  esac
}

run_arm() {
  local arm="$1"
  local run_id="lsweep_qwen06_${arm}_s${SEED}"
  local out="outputs/${run_id}.pt"
  local log="logs/training_runs/${run_id}.log"
  mkdir -p outputs logs/training_runs

  # guard on the ARTIFACT, never the directory -- a killed job leaves an empty dir
  # and a dir-guard makes the rerun skip and the judge read a file that never existed.
  if [ -s "$out" ]; then echo "[skip] $arm -- $out already exists"; return 0; fi

  local extra; extra="$(arm_flags "$arm")"
  echo "=== arm ${arm} :: seed ${SEED} :: ${STEPS} steps :: deltas [${extra:-none}] ==="
  [ "$DRY_RUN" = 1 ] && return 0

  # shellcheck disable=SC2086
  "$PY" -u experiments/train_version_g_final.py \
    --model-id "$MODEL_ID" --out "$out" --run-id "$run_id" \
    --train-scope all --abliterate-layers all --attack-ensemble \
    --attack-profile version_g_final --attack-layers all --direction-layer 20 \
    --vg-rank-k-prob 0.10 --version-g-attack-ranks 1,2,4,8,16 \
    --version-g-rank-estimator arditi_residual \
    --vg-surgical-prob 0.40 --vg-surgical-cap-ranks 2,4,8,16 --vg-heretic-prob 0.35 \
    --no-grad-checkpoint --recompute-direction-every 25 \
    --lambda-rr 4 --rr-center --harm-targets data/harm_targets_qwen.json --rr-layers last_half \
    --lambda-gib 0 --stage2-lambda-gib 0 --gib-mode argmax \
    --lambda-uncensor 4 --uncensor-margin 4 --lambda-harm 4 --harm-margin 4 \
    --lambda-safe 4 --stage2-lambda-safe 4 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 64 --clean-start-step 0 --clean-ramp-steps 100 \
    --refusal-file data/extended_refusals_advbench.json --refusal-max-len 384 \
    --advbench-preview-tokens 0 --gsm8k-probe-n 0 --advbench-judge-n 0 \
    --n-direction 256 --vg-n-capability 256 \
    --steps "$STEPS" --eval-every 0 --save-every "$SAVE_EVERY" --lr 1e-5 --seed "$SEED" \
    --qwen-thinking on \
    $extra 2>&1 | tee "$log"
}

for arm in ${ARMS:-${1:?usage: sweep_lambda_qwen06.sh <arm> | ARMS="a0 a1 ..." ...}}; do
  run_arm "$arm"
done
echo "[done] arms: ${ARMS:-$1}"
