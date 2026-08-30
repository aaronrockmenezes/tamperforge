#!/usr/bin/env bash
# Loss-term ablation matrix. Separate from sweep_lambda_qwen06.sh -- that one perturbs
# lambda VALUES around the control; this one KNOCKS TERMS OUT to find which are load-bearing.
#
# WHY (docs/plan_2026_08_30_ablation_matrix.md):
#   - L_harm is dead in every run measured (6-7 nonzero steps out of 500-1000): harm_margin=4
#     is unreachably low against measured harm_abl of 12-15. c1 fixes it at 16.
#   - rr_layers=last_half never scored the layer the attack actually wins at on Phi
#     (DL=13, rr starts at 16). Phi was the only one of five trained models with DL outside
#     rr_layers, and the only one that demonstrably failed. As of 2026-08-30 EVERY arm runs
#     --rr-layers all (matching --attack-layers all), so the supervision gap is closed for
#     the whole matrix rather than isolated in one arm. c1 is now the harm-margin fix only.
#   - No one has ever established that the attacked branch contributes anything at all.
#     b7 is the floor: every attacked-branch term off. If b7 walls like a0, the wall came
#     from the attack distribution (multi-rank + surgical + all-layer), not the loss.
#
#   MODEL=qwen06 ARMS="a0 b1 b2" bash scripts/runs/ablation_matrix.sh
#   MODEL=phi4mini ARMS="a0 c1" GPU=1 bash scripts/runs/ablation_matrix.sh
#
# DRY_RUN=1 prints the plan and launches nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }

PY="${PY:-/venv/main/bin/python}"
MODEL="${MODEL:?set MODEL: qwen06 | phi4mini | qwen3_4b | gemma4_e2b}"
STEPS="${STEPS:-1000}"
SAVE_EVERY="${SAVE_EVERY:-200}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export PYTHONPATH=src

# Cap torch/OMP/MKL threads when running >1 arm concurrently on one box, one process per GPU.
# Without this, each process defaults to spawning ~nproc threads; N concurrent processes
# fighting over the same cores causes catastrophic context-switch thrashing, not linear
# slowdown -- measured 2026-08-30: 4-way concurrent on a 96-thread box, uncapped, ran ~83x
# slower than solo (878s/it vs 10.6s/it). Capped at nproc/4, both matched solo speed exactly.
NPROC="$(nproc 2>/dev/null || echo 8)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$((NPROC / 4 > 0 ? NPROC / 4 : 1))}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

# ---- per-model config -------------------------------------------------------
# direction_layer values are the ones the 2026-08-15 Pro6000 campaign actually used
# (read back from each run's own training-config line, not guessed). n_layers is from
# the trainable-matrix count in the same logs. "DL vs rr" flags the supervision gap.
case "$MODEL" in
  qwen06)     MODEL_ID=Qwen/Qwen3-0.6B                          ; DL=20 ; THINK=on  ;;  # n=28 rr>=14 DL inside
  phi4mini)   MODEL_ID=microsoft/Phi-4-mini-instruct             ; DL=13 ; THINK=off ;;  # n=32 rr>=16 DL OUTSIDE
  qwen3_4b)   MODEL_ID=Qwen/Qwen3-4B-Instruct-2507               ; DL=19 ; THINK=off ;;  # n=36 rr>=18 DL inside
  gemma4_e2b) MODEL_ID=google/gemma-4-E2B-it                     ; DL=17 ; THINK=off ;;  # n=35 rr>=17 DL inside
  *) echo "unknown MODEL: $MODEL" >&2; exit 2 ;;
esac
export TF_QWEN_THINKING="$THINK"

# ---- arms: ONLY the deltas from the control ---------------------------------
# a0 is the current recipe verbatim. b* knock terms out. c1 applies both fixes.
arm_flags() {
  case "$1" in
    a0) echo "" ;;
    b1) echo "--lambda-uncensor 0" ;;                              # let attacked model refuse
    b2) echo "--lambda-rr 0" ;;                                    # is rerouting doing anything?
    b3) echo "--lambda-clean 0" ;;                                 # what is clean-preservation costing?
    b4) echo "--lambda-uncensor 0 --lambda-rr 0" ;;                # refusal-only defense
    b5) echo "--lambda-uncensor 0 --lambda-clean 0" ;;
    b6) echo "--lambda-rr 0 --lambda-clean 0" ;;
    b7) echo "--lambda-uncensor 0 --lambda-rr 0 --lambda-clean 0" ;; # floor: attacked branch off
    c1) echo "--harm-margin 16" ;;                                 # rr-layers all is now baseline
    e1) echo "--harm-margin 16" ;;                                 # SAME recipe as c1, separate
                                                                    # run-id so it doesn't collide
                                                                    # with c1's finished checkpoint.
                                                                    # Launch with SAVE_EVERY=100
                                                                    # (only for this arm) for a
                                                                    # trajectory view of the round-1
                                                                    # winner: SAVE_EVERY=100 MODEL=qwen06
                                                                    # ARMS=e1 bash ...
    c2) echo "--lambda-harm 0" ;;                                  # drop the dead term entirely --
                                                                    # QUEUED, not part of the first
                                                                    # 9-arm pass; run after c1 lands
    # --- 2x2x2 factorial over {rr, uncensor, harm}, lambda_clean fixed at baseline (3) in
    # EVERY arm below, unlike b3/b5/b6/b7 which zeroed it. a0/b1/b2/b4 already cover 4 of the
    # 8 cells (rr,uncensor,harm all "on" except the one/two named); d1-d3 fill the rest.
    d1) echo "--lambda-uncensor 0 --lambda-harm 0" ;;               # rr:on  unc:off harm:off
    f1) echo "--lambda-uncensor 0 --lambda-harm 0" ;;               # SAME recipe as d1 (the round-2
                                                                    # champion, PPS 0.938), separate
                                                                    # run-id so it doesn't collide
                                                                    # with d1's finished checkpoint.
                                                                    # Launch with SAVE_EVERY=100 for
                                                                    # a trajectory view: SAVE_EVERY=100
                                                                    # MODEL=qwen06 ARMS=f1 bash ...
    d2) echo "--lambda-rr 0 --lambda-harm 0" ;;                     # rr:off unc:on  harm:off
    d3) echo "--lambda-rr 0 --lambda-uncensor 0 --lambda-harm 0" ;; # rr:off unc:off harm:off (floor, clean intact)
    *)  echo "unknown arm: $1" >&2; exit 2 ;;
  esac
}

run_arm() {
  local arm="$1"
  local run_id="abl_${MODEL}_${arm}_s${SEED}"
  local out="outputs/${run_id}.pt"
  local log="logs/training_runs/${run_id}.log"
  mkdir -p outputs logs/training_runs

  # guard on the ARTIFACT, not the directory: a killed job leaves an empty dir and a
  # dir-guard makes the rerun skip, then the judge reads a file that never existed.
  if [ -s "$out" ]; then echo "[skip] $arm -- $out exists"; return 0; fi

  local extra; extra="$(arm_flags "$arm")"
  echo "=== ${MODEL} :: ${arm} :: DL=${DL} :: ${STEPS} steps :: [${extra:-control}] ==="
  [ "$DRY_RUN" = 1 ] && return 0

  # shellcheck disable=SC2086
  "$PY" -u experiments/train_version_g_final.py \
    --model-id "$MODEL_ID" --out "$out" --run-id "$run_id" \
    --train-scope all --abliterate-layers all --attack-ensemble \
    --attack-profile version_g_final --attack-layers all --direction-layer "$DL" \
    --vg-rank-k-prob 0.10 --version-g-attack-ranks 1,2,4,8,16 \
    --version-g-rank-estimator arditi_residual \
    --vg-surgical-prob 0.40 --vg-surgical-cap-ranks 2,4,8,16 --vg-heretic-prob 0.35 \
    --no-grad-checkpoint --recompute-direction-every 25 \
    --grad-clip 1.0 \
    --lambda-rr 4 --rr-center --harm-targets data/harm_targets_qwen.json --rr-layers all \
    --lambda-gib 0 --stage2-lambda-gib 0 --gib-mode argmax \
    --lambda-uncensor 4 --uncensor-margin 4 --lambda-harm 4 --harm-margin 4 \
    --lambda-safe 4 --stage2-lambda-safe 4 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 64 --clean-start-step 0 --clean-ramp-steps 100 \
    --refusal-file data/extended_refusals_advbench.json --refusal-max-len 384 \
    --advbench-preview-tokens 0 --gsm8k-probe-n 0 --advbench-judge-n 0 \
    --n-direction 256 --vg-n-capability 256 \
    --steps "$STEPS" --eval-every 0 --save-every "$SAVE_EVERY" --lr 1e-5 --seed "$SEED" \
    --qwen-thinking "$THINK" \
    $extra 2>&1 | tee "$log"
}

echo "model=${MODEL} (${MODEL_ID}) DL=${DL} thinking=${THINK} steps=${STEPS} seed=${SEED} gpu=${GPU}"
for arm in ${ARMS:-a0 b1 b2 b3 b4 b5 b6 b7 c1}; do run_arm "$arm"; done
echo "[done] ${MODEL}: ${ARMS:-full matrix}"
