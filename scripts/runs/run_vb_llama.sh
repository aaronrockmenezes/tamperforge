#!/usr/bin/env bash
# version_B on Llama-3.2-1B-Instruct. Architecture replication of the poison-pill claim.
#
# Why version_B and not A or C: version_B is the only version where BOTH rank-1 AND surgical
# ablation are self-defeating (Qwen: GSM8K 0.3889 -> 0.0091 and -> 0.1054). version_A blocks
# surgical inertly (capability intact), version_C broke on it. The unoccupied cell in the
# literature is abliteration x capability-collapse, so that is the claim worth replicating.
#
# --direction-layer 13 is MEASURED, not scaled: the base ablation sweep across all 16 layers
# peaks at L13 (0.5650 judged harmful). L11 -- what proportional scaling from Qwen's 20/28
# would have given -- is a local MINIMUM at 0.1950, and L15 is dead at 0.0700.
# devlog_2026_07_02 records a mid-depth guess (L8) costing 0.654 -> 0.133 until corrected.
#
# --attack-layers 6-14 spans the sustained band; note version_B's sampler derives its own
# bands from n_layers and ignores this flag, which is only here to satisfy validation.
#
# --no-grad-checkpoint, matching version_B on Qwen exactly so the recipe is identical across
# architectures. Measured: 12.8GB peak of 24GB, 4.13 s/it -- same speed as checkpointed at
# this scale, so checkpointing was buying nothing.
#
# Llama reference frame (base, same export path):
#   clean 0.0019 | rank-1 0.6288 | surgical 0.6923 | heretic 0.8269  <- ceiling
#   span 0.8250 vs Qwen's 0.4385, and heretic is the APEX attack here (it was not on Qwen)
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs outputs

python -u experiments/train_tamper_resistant_v8.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --attack-profile version_b --attack-ensemble --attack-layers 6-14 \
  --version-a-p-canonical 0.2 --version-a-p-surgical 0.4 \
  --version-a-cap-ranks 2,4,8,16 --version-a-n-cap 256 \
  --steps 500 --save-every 25 --eval-every 25 \
  --clean-start-step 250 --clean-ramp-steps 100 \
  --stage2-lambda-gib 8.0 --stage2-lambda-safe 4.0 \
  --lambda-clean 3.0 --lambda-gib 8.0 --lambda-reg 0.1 \
  --lambda-safe 1.0 --lambda-uncensor 4.0 \
  --direction-layer 13 --recompute-direction-every 25 \
  --train-scope all --no-grad-checkpoint \
  --ifeval-in-loop --ifeval-probe-n 24 \
  --qwen-thinking off --seed 42 \
  --out outputs/version_b_llama_500.pt
echo "TRAIN_RC=$?"
