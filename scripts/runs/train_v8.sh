#!/usr/bin/env bash
# Standard ABL-v8 two-stage train (generative clean-anchor + curriculum + save-every 4-axis pick).
# Any model. gemma needs eager attn (auto via load_model). After: run pick_v8_best.sh, judge, pick.
#   MODEL=google/gemma-3-1b-it DL=13 OUT=outputs/tamper_resistant_gemma3_1b_v8.pt \
#     CUDA_VISIBLE_DEVICES=0 bash scripts/train_v8.sh
# Overridable: LGIB(8) LCLEAN(3) S2GIB(4) S2SAFE(4) SEED(42) STEPS(500). Llama: keep S2SAFE=4.
# ABL-v9 (all optional, unset = exact v8): LHARM LRR HARMTGT PARTIAL=1 PERLAYER=1
# PY overrides the interpreter (boxes have no bare `python` on PATH; same fix as pick_v8_best.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MODEL="${MODEL:?set MODEL}"; DL="${DL:?set DL (judged refusal-layer peak)}"; OUT="${OUT:?set OUT}"
"${PY:-python}" experiments/train_tamper_resistant_v8.py --model-id "$MODEL" --out "$OUT" \
  --train-scope all --abliterate-layers all --attack-ensemble --direction-layer "$DL" \
  --recompute-direction-every 25 --gib-mode argmax --gib-gen-tokens 32 --gib-gen-prompts 2 \
  --lambda-gib "${LGIB:-8}" --lambda-uncensor 4 --lambda-safe 1 --lambda-reg 0.1 \
  --lambda-clean "${LCLEAN:-3}" --clean-gen-prompts 2 --clean-gen-tokens 32 \
  --clean-start-step 250 --clean-ramp-steps 100 \
  --stage2-lambda-gib "${S2GIB:-4}" --stage2-lambda-safe "${S2SAFE:-4}" \
  --ifeval-in-loop --ifeval-probe-n 24 --save-every 25 \
  --steps "${STEPS:-500}" --eval-every 25 --lr 1e-5 --seed "${SEED:-42}" \
  ${LHARM:+--lambda-harm "$LHARM"} ${LRR:+--lambda-rr "$LRR"} \
  ${HARMTGT:+--harm-targets "$HARMTGT"} \
  ${PARTIAL:+--attack-partial} ${PERLAYER:+--attack-per-layer}
echo "### trained -> $OUT (+ .s*.pt snapshots). Next: prune stage-1 (.s25..s225), then"
echo "  MID=$MODEL STEM=$OUT DL=$DL bash scripts/pick_v8_best.sh  -> judge -> 4-axis pick ###"
