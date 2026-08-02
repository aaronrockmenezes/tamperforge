#!/usr/bin/env bash
# ART baseline -- abliteration-resistant tuning (Kuo, Yadav, Smith, arXiv:2605.26526).
# The other must-baseline competitor in docs/related_work.md, and the one occupying our own
# cell: they simulate worst-case abliteration on the current parameters and do gradient
# ascent on harmful outputs, needing no data beyond the existing alignment set.
#
# related_work.md already identifies ART as "our v9 idea-1 (harmful-side loss under attack)
# plus idea-8 (inner-loop attack search), already published", so it maps onto flags we have:
#
#   --attack-ensemble --attack-profile version_b   inner-loop attack search (idea-8)
#   --lambda-uncensor 4                            refusal must SURVIVE ablation
#   --lambda-harm 4                                attacked model must not COMPLY (idea-1)
#   --lambda-gib 0 --stage2-lambda-gib 0           NO capability collapse -- that is OURS,
#                                                  not theirs. ART is a fortress by design.
#
# Everything else matches version_B so the only variables are the objective terms. Canned
# one-line REFUSAL_RESPONSES (extended refusals are Shairah's contribution, not ART's).
#
# --save-every 500: only the final checkpoint. 61GB free and 20 intermediates per run would
# be 17GB (qwen) + 38GB (llama).
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs outputs

if [ -f outputs/art_qwen_500.pt ]; then
  echo "[skip] qwen already trained"
else
  echo "=== ART qwen $(date -u) ==="
  python -u experiments/train_tamper_resistant_v8.py \
    --model-id Qwen/Qwen3-0.6B --out outputs/art_qwen_500.pt \
    --train-scope all --abliterate-layers all --attack-ensemble \
    --attack-profile version_b --attack-layers 10-27 --direction-layer 20 \
    --no-grad-checkpoint --recompute-direction-every 25 \
    --lambda-gib 0 --stage2-lambda-gib 0 \
    --lambda-uncensor 4 --uncensor-margin 4 \
    --lambda-harm 4 --harm-margin 4 \
    --lambda-safe 1 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 32 \
    --clean-start-step 250 --clean-ramp-steps 100 --stage2-lambda-safe 4 \
    --ifeval-in-loop --ifeval-probe-n 24 \
    --n-direction 256 --version-a-n-cap 256 \
    --steps 500 --eval-every 25 --save-every 500 --lr 1e-5 --seed 42 \
    --qwen-thinking off \
    2>&1 | tee logs/training_runs/art_qwen_500.log
  echo "QWEN_RC=${PIPESTATUS[0]}"
fi

if [ -f outputs/art_llama_500.pt ]; then
  echo "[skip] llama already trained"
else
  echo "=== ART llama $(date -u) ==="
  python -u experiments/train_tamper_resistant_v8.py \
    --model-id meta-llama/Llama-3.2-1B-Instruct --out outputs/art_llama_500.pt \
    --train-scope all --abliterate-layers all --attack-ensemble \
    --attack-profile version_b --attack-layers 6-14 --direction-layer 13 \
    --no-grad-checkpoint --recompute-direction-every 25 \
    --lambda-gib 0 --stage2-lambda-gib 0 \
    --lambda-uncensor 4 --uncensor-margin 4 \
    --lambda-harm 4 --harm-margin 4 \
    --lambda-safe 1 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 32 \
    --clean-start-step 250 --clean-ramp-steps 100 --stage2-lambda-safe 4 \
    --ifeval-in-loop --ifeval-probe-n 24 \
    --n-direction 256 --version-a-n-cap 256 \
    --steps 500 --eval-every 25 --save-every 500 --lr 1e-5 --seed 42 \
    --qwen-thinking off \
    2>&1 | tee logs/training_runs/art_llama_500.log
  echo "LLAMA_RC=${PIPESTATUS[0]}"
fi

echo "=== ART training DONE $(date -u) ==="
ls -lh outputs/art_*.pt 2>/dev/null
df -h /workspace | tail -1
