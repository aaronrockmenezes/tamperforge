#!/usr/bin/env bash
# version_F -- stop iterating our own objective; compose the two baselines that each beat us
# on one axis, and drop the term that costs us the other.
#
# WHY THIS SHAPE (all four inputs are measured, none are guesses):
#
# 1. DROP lambda_gib ENTIRELY. gib_ce at step 500 vs MT-Bench is monotone inverse across the
#    whole E series: E1 3.05/3.04, version_B high/3.52, E2 0.49/4.46. And it does not pay for
#    itself -- MAD fires on READ-projection ablation, heretic ablates write projections only
#    and never triggers it (CLAUDE.md, 2026-08-02). It is the term that breaks speech in
#    exchange for nothing on the only gate nothing passes. So version_F is NOT a MAD run.
#
# 2. ART's harm-side objective (--lambda-uncensor 4 --lambda-harm 4). ART scored MT-Bench
#    4.325 / heretic 0.7058 -- it fails both gates on its own, but it is the only objective in
#    the repo that pressures the ATTACKED model not to comply, rather than pressuring it to
#    babble.
#
# 3. --clean-start-step 0. The single flag separating E1 from E2, worth +1.42 MT-Bench
#    (3.04 -> 4.46) and XSTest 0.428 -> 0.624. ART ran with 250, i.e. 250 steps of no clean
#    pressure at all. This is the cheapest MT-Bench point on the table.
#
# 4. Shairah's extended refusals. Spreading the refusal signal over many token positions is
#    mechanically orthogonal to ART's "refusal must survive a simulated ablation" -- one
#    changes the target text, the other changes what the target is robust to. Neither passes
#    gate 2 alone (Shairah 0.5365-0.7865, ART 0.7058). Whether they compose is unmeasured, and
#    it is the only untested combination left that does not require a new sampler.
#
# WHAT IS DELIBERATELY NOT HERE: a new attack sampler. Sampling has failed fixed (v8), widened
# (version_A/B) and adaptive-against-a-live-optimiser (version_C, which trained write-only ~60%
# of steps and regressed). Measured on version_B's sampler, write-only is 5.9% of steps and
# heretic's actual shape (write-only + near-full-stack) is 0.04% -- but version_C already
# bought that coverage and it did not place the entanglement. Structural, not a coverage gap.
#
# THE HONEST PREDICTION, recorded before the run so the result is falsifiable either way:
# gate 1 PASSES (ART 4.325 + clean-start-0's ~1.4 puts it comfortably over 4.46) and gate 2
# FAILS around 0.70, because both gate-2 ingredients failed gate 2 alone. The reason to run it
# anyway is that "do ART and Shairah compose against heretic?" has no answer in the literature
# or in this repo, and a clean NO is a paper paragraph. If gate 2 passes, it is the final
# version.
#
# NOTE ON THE GATE-2 BAR. Nothing has ever passed gate 2 without being brain-damaged first:
# version_B 3.52/0.3212, ART 4.325/0.7058, E2 4.46/0.7308, base 4.96/0.6788. If version_F lands
# on that line it tells us gate 2 is measuring competence, not defence, and the gates need
# rethinking before another training run -- write that up, do not tune around it.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs outputs logs/probes

TAG=version_f_qwen_500
REF=data/extended_refusals_advbench.json
[ -f "$REF" ] || { echo "MISSING $REF"; exit 1; }

say () { echo "[$(date -u +%H:%M:%S)] $*"; }

# vLLM cannot share the GPU with a trainer. tmux has-session PREFIX-matches, so match exactly
# or a chain waits on itself.
for S in artfin artchain arqher arlher serve_eval; do
  if tmux ls -F "#{session_name}" 2>/dev/null | grep -qx "$S"; then
    say "[wait] $S running..."
    while tmux ls -F "#{session_name}" 2>/dev/null | grep -qx "$S"; do sleep 60; done
    say "[wait] $S done, settling 30s"; sleep 30
  fi
done

if [ -f "outputs/${TAG}.pt" ]; then
  say "[skip] $TAG already trained"
else
  say "=== TRAIN $TAG ==="
  python -u experiments/train_tamper_resistant_v8.py \
    --model-id Qwen/Qwen3-0.6B --out "outputs/${TAG}.pt" \
    --train-scope all --abliterate-layers all --attack-ensemble \
    --attack-profile version_b --attack-layers 10-27 --direction-layer 20 \
    --no-grad-checkpoint --recompute-direction-every 25 \
    --lambda-gib 0 --stage2-lambda-gib 0 \
    --lambda-uncensor 4 --uncensor-margin 4 \
    --lambda-harm 4 --harm-margin 4 \
    --lambda-safe 1 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 64 \
    --clean-start-step 0 --clean-ramp-steps 100 --stage2-lambda-safe 4 \
    --refusal-file "$REF" --refusal-max-len 384 \
    --ifeval-in-loop --ifeval-probe-n 24 \
    --gsm8k-probe-n 8 --gsm8k-probe-max-new 256 \
    --n-direction 256 --version-a-n-cap 256 \
    --steps 500 --eval-every 25 --save-every 500 --lr 1e-5 --seed 42 \
    --qwen-thinking off \
    2>&1 | tee "logs/training_runs/${TAG}.log"
  say "  ${TAG}_RC=${PIPESTATUS[0]}"
fi

[ -f "outputs/${TAG}.pt" ] || { say "[FAIL] no checkpoint, stopping"; exit 1; }

# Materialise clean and run the 5 prompts before spending an hour on gate 1.
HF="outputs/${TAG}_clean"
[ -f "$HF/model.safetensors" ] || python -u experiments/save_p1b_checkpoint.py \
  --checkpoint "outputs/${TAG}.pt" --model-id Qwen/Qwen3-0.6B --attack none \
  --out "$HF" >>"logs/training_runs/${TAG}.log" 2>&1
[ -f "$HF/model.safetensors" ] || { say "[FAIL] could not materialise $HF"; exit 1; }

say "  --- smoke5 gate: $TAG ---"
python scripts/probes/smoke5.py "$HF" --max-new 120 --modes nothink,default \
  2>&1 | tee "logs/probes/smoke5_${TAG}.log" | grep -aE "^\[|^MODEL|^====" | head -30

say "=== version_F TRAINING DONE -- next is GATE 1 (MT-Bench), run it separately ==="
ls -lh "outputs/${TAG}.pt"
df -h /workspace | tail -1
