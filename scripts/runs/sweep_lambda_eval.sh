#!/usr/bin/env bash
# Fresh-attack evaluation for the lambda sweep arms. Two-stage, to keep the cost sane.
#
#   stage A: every decoder layer, 16 prompts   -> where is the attacker's peak?
#   stage B: top-3 layers from A, 64 prompts   -> confirm the rate there
#
# Stage A is exact for argmax and ~4x cheaper than sweeping every layer at 64. Only the
# confirmed stage-B rate is ever reported. Both stages re-estimate the refusal basis on the
# checkpoint under test -- the protocol from docs/findings_fresh_rank_attacks_2026_08_15.md.
#
#   bash scripts/runs/sweep_lambda_eval.sh a0
#   ARMS="base a0 a1 a2 a3 a4 a5" bash scripts/runs/sweep_lambda_eval.sh
#
# 'base' evaluates the untouched base model -- this is the reference line that does not
# currently exist for Qwen and that the stopping rule's criterion 5 needs.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }

PY="${PY:-/venv/main/bin/python}"
MODEL_ID=Qwen/Qwen3-0.6B
SEED="${SEED:-42}"
RANK="${RANK:-1}"
GPU="${GPU:-0}"
TOPK="${TOPK:-3}"
# Run-id prefix the checkpoints were saved under. sweep_lambda_qwen06.sh uses lsweep_qwen06_;
# ablation_matrix.sh uses abl_qwen06_. Override with PREFIX=abl_qwen06 to match.
PREFIX="${PREFIX:-lsweep_qwen06}"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export TF_QWEN_THINKING=on
export PYTHONPATH=src

sweep() {   # sweep <outdir> <n_harmful> <ckpt_args...> [--layers ...]
  local out="$1"; shift
  local n="$1"; shift
  mkdir -p "$out/generations"
  "$PY" -u scripts/probes/adaptive_attack_sweep.py \
    --model-id "$MODEL_ID" "$@" \
    --attack-ranks "$RANK" --variants "rank${RANK}" \
    --rank-estimator arditi_residual \
    --n-direction 64 --n-capability 64 --n-harmful "$n" \
    --direction-batch-size 8 --generation-batch-size 64 \
    --snapshot-device model --max-new-tokens 1024 \
    --judge-workers 64 --judge-max-tokens 1024 --judge-timeout-seconds 600 \
    --out "$out/summary.json" --gen-dir "$out/generations"
}

top_layers() {   # read stage-A summary, emit top-K layers by the canonical rule
  "$PY" - "$1" "$TOPK" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
rows = next(iter(s["variants"].values()))
# canonical selection: max judged harm, then MIN gibberish, then MIN refused.
# a plain max() over tuples tiebreaks on MAX gibberish and picks a different layer.
rows.sort(key=lambda r: (-r["harmful_actionable"], r["gibberish"], r.get("refused", 0.0)))
print(",".join(str(r["layer"]) for r in rows[: int(sys.argv[2])]))
PYEOF
}

eval_arm() {
  local arm="$1"
  local tag="${PREFIX}_${arm}_s${SEED}_r${RANK}"
  local ckpt=(--checkpoint "outputs/${PREFIX}_${arm}_s${SEED}.pt")
  [ "$arm" = base ] && ckpt=()
  [ "$arm" = base ] || [ -s "outputs/${PREFIX}_${arm}_s${SEED}.pt" ] || {
    echo "[skip] $arm -- no checkpoint yet"; return 0; }

  local a="results/dl_sweeps/${tag}_stageA"
  local b="results/dl_sweeps/${tag}_stageB"

  if [ -s "$a/summary.json" ]; then echo "[skip A] $arm"; else
    echo "=== $arm stage A :: all layers @16 ==="
    sweep "$a" 16 "${ckpt[@]}"
  fi

  if [ -s "$b/summary.json" ]; then echo "[skip B] $arm"; else
    local L; L="$(top_layers "$a/summary.json")"
    echo "=== $arm stage B :: layers ${L} @64 ==="
    sweep "$b" 64 "${ckpt[@]}" --layers "$L"
  fi
}

for arm in ${ARMS:-${1:?usage: sweep_lambda_eval.sh <arm> | ARMS="base a0 ..." ...}}; do
  eval_arm "$arm"
done
echo "[done] eval arms: ${ARMS:-$1}"
