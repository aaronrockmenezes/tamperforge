#!/usr/bin/env bash
# Shairah et al. extended-refusal baseline (arXiv:2505.19056), Qwen3-0.6B + Llama-3.2-1B.
# TODO.md [HIGH] -- the single biggest reviewer risk in the project.
#
# Their claim: fine-tuning on EXTENDED refusals (neutral overview -> explicit refusal ->
# ethical rationale) spreads the refusal signal across many token positions, so abliteration
# cannot strip it with a rank-1 edit. No adversarial training, no attack simulation, no
# inner loop. They report refusal dropping at most 10% under abliteration vs 70-80% for
# conventional safety tuning.
#
# Our REFUSAL_RESPONSES are one-liners -- exactly the conventional tuning they beat. So this
# arm is version_B's recipe with two changes and nothing else:
#   1. --refusal-file swaps the one-liners for extended refusals (520, mean 143.8 words,
#      max 236 tokens; --refusal-max-len 384 keeps the ethical rationale off the truncator).
#   2. every adversarial term OFF: --lambda-gib 0 --stage2-lambda-gib 0 --lambda-uncensor 0.
#      Both gib lambdas must be 0 -- `gib_active` ORs them, and a nonzero either side
#      re-enables the generation loop that IS the MAD mechanism.
# --lambda-clean/--lambda-safe/--lambda-reg and the schedule stay at version_B's values so
# the only variables are the refusal text and the adversarial machinery.
#
# What this must produce (TODO.md): capability under attack, which their paper does not
# report. If extended-refusal also craters GSM8K under abliteration, our differentiator is
# gone. If refusal holds AND capability survives, that is the clean fortress/poison-pill
# split and belongs in the paper's first table.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs outputs

REF=data/extended_refusals_advbench.json
[ -f "$REF" ] || { echo "MISSING $REF -- run experiments/gen_extended_refusals.py"; exit 1; }

# Training wants the GPU to itself (12.8GB peak); the xstest matrix is holding vLLM at
# 0.45 util. Wait it out rather than racing it.
if tmux has-session -t xstest 2>/dev/null; then
  echo "[wait] xstest matrix still running, polling..."
  while tmux has-session -t xstest 2>/dev/null; do sleep 60; done
  echo "[wait] xstest done, settling 30s"; sleep 30
fi

# --refusal-file and --refusal-max-len are new code. Prove the path end-to-end on 6 steps
# before spending ~45min per architecture.
if [ ! -f /tmp/shairah_smoke_ok ]; then
  echo "=== smoke (6 steps, qwen) $(date -u) ==="
  python -u experiments/train_tamper_resistant_v8.py \
    --model-id Qwen/Qwen3-0.6B --out /tmp/shairah_smoke.pt \
    --refusal-file "$REF" --refusal-max-len 384 \
    --train-scope all --abliterate-layers all \
    --direction-layer 20 --no-grad-checkpoint \
    --lambda-gib 0 --stage2-lambda-gib 0 --lambda-uncensor 0 \
    --lambda-safe 1 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 1 --clean-gen-tokens 8 \
    --n-direction 32 --refusal-batch 2 \
    --steps 6 --eval-every 1000 --save-every 1000 --lr 1e-5 --seed 42 \
    --qwen-thinking off > logs/training_runs/shairah_smoke.log 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== SMOKE FAILED rc=$rc -- not training. tail: ==="
    tail -25 logs/training_runs/shairah_smoke.log
    exit 1
  fi
  grep -a "extended refusals:" logs/training_runs/shairah_smoke.log
  rm -f /tmp/shairah_smoke.pt; touch /tmp/shairah_smoke_ok
  echo "=== smoke OK $(date -u) ==="
fi

# ---------------- Qwen3-0.6B (version_B qwen recipe, DL 20) ----------------
if [ -f outputs/shairah_qwen_500.pt ]; then
  echo "[skip] qwen already trained"
else
  echo "=== shairah qwen $(date -u) ==="
  python -u experiments/train_tamper_resistant_v8.py \
    --model-id Qwen/Qwen3-0.6B --out outputs/shairah_qwen_500.pt \
    --refusal-file "$REF" --refusal-max-len 384 \
    --train-scope all --abliterate-layers all \
    --direction-layer 20 --recompute-direction-every 25 --no-grad-checkpoint \
    --lambda-gib 0 --stage2-lambda-gib 0 --lambda-uncensor 0 \
    --lambda-safe 1 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 32 \
    --clean-start-step 250 --clean-ramp-steps 100 --stage2-lambda-safe 4 \
    --ifeval-in-loop --ifeval-probe-n 24 \
    --n-direction 256 \
    --steps 500 --eval-every 25 --save-every 25 --lr 1e-5 --seed 42 \
    --qwen-thinking off \
    2>&1 | tee logs/training_runs/shairah_qwen_500.log
  echo "QWEN_RC=${PIPESTATUS[0]}"
fi

# ---------------- Llama-3.2-1B (version_B llama recipe, DL 13) ----------------
if [ -f outputs/shairah_llama_500.pt ]; then
  echo "[skip] llama already trained"
else
  echo "=== shairah llama $(date -u) ==="
  python -u experiments/train_tamper_resistant_v8.py \
    --model-id meta-llama/Llama-3.2-1B-Instruct --out outputs/shairah_llama_500.pt \
    --refusal-file "$REF" --refusal-max-len 384 \
    --train-scope all --abliterate-layers all \
    --direction-layer 13 --recompute-direction-every 25 --no-grad-checkpoint \
    --lambda-gib 0 --stage2-lambda-gib 0 --lambda-uncensor 0 \
    --lambda-safe 1 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 32 \
    --clean-start-step 250 --clean-ramp-steps 100 --stage2-lambda-safe 4 \
    --ifeval-in-loop --ifeval-probe-n 24 \
    --n-direction 256 \
    --steps 500 --eval-every 25 --save-every 25 --lr 1e-5 --seed 42 \
    --qwen-thinking off \
    2>&1 | tee logs/training_runs/shairah_llama_500.log
  echo "LLAMA_RC=${PIPESTATUS[0]}"
fi

echo "=== shairah training DONE $(date -u) ==="
ls -l outputs/shairah_*.pt
