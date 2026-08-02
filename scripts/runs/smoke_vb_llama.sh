#!/usr/bin/env bash
# 6-step smoke of version_B on Llama-3.2-1B before committing 2h.
# Checks: (1) it fits in 24GB with grad checkpointing on, (2) the version_B sampler adapts to
# 16 layers, (3) nothing else is Qwen-only. heretic_directions was Qwen-only until today
# (Llama's tokeniser has no pad token), so architecture assumptions are not hypothetical here.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
python -u experiments/train_tamper_resistant_v8.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --attack-profile version_b --attack-ensemble --attack-layers 6-14 \
  --version-a-p-canonical 0.2 --version-a-p-surgical 0.4 \
  --version-a-cap-ranks 2,4,8,16 --version-a-n-cap 256 \
  --steps 6 --save-every 0 --eval-every 6 \
  --clean-start-step 250 --clean-ramp-steps 100 \
  --stage2-lambda-gib 8.0 --stage2-lambda-safe 4.0 \
  --lambda-clean 3.0 --lambda-gib 8.0 --lambda-reg 0.1 \
  --lambda-safe 1.0 --lambda-uncensor 4.0 \
  --direction-layer 13 --recompute-direction-every 25 \
  --train-scope all --no-grad-checkpoint \
  --qwen-thinking off --seed 42 \
  --out outputs/vb_llama_smoke.pt
echo "SMOKE_RC=$?"
