#!/usr/bin/env bash
# version_C full run. Identical to version_B's recipe except the attack profile,
# so any difference in the result is attributable to the attack distribution.
#
# Deviation from version_B, deliberate: --version-a-p-surgical 0.20 (was 0.40).
# sample_attack_c splits the remaining mass between the heretic buffer and random
# tents, and at 0.40 surgical those two get squeezed to 0.30/0.10. At 0.20 the mix
# is canonical .20 / surgical .20 / buffer .30 / tent .30. Surgical robustness is a
# hard-won version_B result, so this is the axis to watch in the eval -- if surgical
# regresses from 0.0000 harmful / 0.98 gibberish, this is why.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs outputs

python -u experiments/train_tamper_resistant_v8.py \
  --model-id Qwen/Qwen3-0.6B \
  --attack-profile version_c --attack-ensemble --attack-layers 10-27 \
  --heretic-every 100 --heretic-trials 24 --heretic-startup-trials 8 \
  --heretic-kl-max 0.5 --heretic-buffer-cap 24 \
  --version-a-p-canonical 0.2 --version-a-p-surgical 0.20 \
  --version-a-cap-ranks 2,4,8,16 --version-a-n-cap 256 \
  --steps 500 --save-every 25 --eval-every 25 \
  --clean-start-step 250 --clean-ramp-steps 100 \
  --stage2-lambda-gib 8.0 --stage2-lambda-safe 4.0 \
  --lambda-clean 3.0 --lambda-gib 8.0 --lambda-reg 0.1 \
  --lambda-safe 1.0 --lambda-uncensor 4.0 \
  --direction-layer 20 --recompute-direction-every 25 \
  --train-scope all --no-grad-checkpoint \
  --ifeval-in-loop --ifeval-probe-n 24 \
  --qwen-thinking off --seed 42 \
  --out outputs/version_c_qwen_500.pt
echo "TRAIN_RC=$?"
