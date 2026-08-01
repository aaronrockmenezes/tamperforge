#!/usr/bin/env bash
# version_D = version_B's recipe with ONE change: --gib-mode task.
#
# Every version so far (v8, A, B, C) varied the ATTACK distribution and left the objective
# alone. The objective asked the attacked model to pick different tokens from the clean model
# over 32 tokens of benign text -- satisfiable without losing capability, and version_C proved
# it (0.94 gibberish on harmful prompts, GSM8K still 0.3715; gib_ce AUC 0.460 = chance).
# --gib-mode task instead scores the attacked model against KNOWN-CORRECT answers, so the only
# way to satisfy it is to actually get the task wrong.
#
# Built on version_B, not version_C: version_B is the better artifact and this isolates the
# objective change from the in-loop machinery.
#
# CONTAMINATION: trains on GSM8K train. lm_eval scores GSM8K test (disjoint), but GSM8K is no
# longer an independent capability measure here -- READ ARC AND MMLU.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs outputs

echo "=== version_D (running concurrent with replication lanes) ==="
# gate removed: running concurrently with the replication lanes
sleep 30

# gap_target is scale-dependent and 4.0 was tuned for a different quantity. Calibrate, or the
# margin either never satisfies (constant max pressure -> the FTR-v6 lobotomy) or satisfies
# from step 1 (trains nothing).
echo "=== calibrating gap_target ===" | tee logs/training_runs/version_d_calib.log
python -u calib_task_gib.py 2>&1 | tee -a logs/training_runs/version_d_calib.log
GT=$(grep -a "SUGGESTED --gap-target" logs/training_runs/version_d_calib.log | awk '{print $NF}')
[ -z "$GT" ] && GT=4.0
echo "=== using --gap-target ${GT} ===" | tee -a logs/training_runs/version_d_calib.log

python -u experiments/train_tamper_resistant_v8.py \
  --model-id Qwen/Qwen3-0.6B \
  --attack-profile version_b --attack-ensemble --attack-layers 10-27 \
  --gib-mode task --gib-task-n 4 --gap-target "$GT" \
  --version-a-p-canonical 0.2 --version-a-p-surgical 0.4 \
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
  --out outputs/version_d_qwen_500.pt
echo "TRAIN_RC=$?"
