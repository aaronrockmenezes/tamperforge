#!/usr/bin/env bash
# version_F -- stop iterating our own objective; compose the two baselines that each beat us
# on one axis, and drop the term that costs us the other.
#
# WHY THIS SHAPE (all four inputs are measured, none are guesses):
#
# 1. lambda_gib 2, NOT 0. The original plan was 0 on the argument that gib_ce buys nothing
#    against heretic (MAD fires on READ-projection ablation; heretic is write-only). That
#    argument is now weaker: MT-Bench on the heretic'd version_B is 2.96, BELOW its own clean
#    3.33, so version_B's heretic resistance runs THROUGH generative degradation rather than
#    alongside it. gib_ce may be load-bearing for the only gate-2 pass we have, which makes
#    setting it to 0 a bet rather than a saving.
#    So probe the middle instead of the endpoint. gib_ce@500 vs MT-Bench is monotone inverse
#    across the E series (E1 3.05/2.74, version_B high/3.33, E2 0.49/4.28); 2 sits between E2's
#    4 and 0, and the informative outcome is whether the trade is CONTINUOUS. If gib 2 lands
#    mid-way on both axes, there is a Pareto curve to optimise and a value worth searching. If
#    it snaps to one end, the trade is a switch and no middle value exists -- which is the
#    finding, and it kills the whole "tune lambda_gib" direction in one run.
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
# 5. THE ATTACK MIX: --version-a-p-canonical 0.10 --version-b-p-heretic 0.35. Arditi drops from
#    20% to 10% and an explicit Heretic-SHAPED slice takes 35% -- write-only (o_proj, down_proj),
#    independent per-projection tents. Measured over 200k draws at 28 layers this moves
#    write-only from 5.9% of steps to 39.1%, where Heretic is write-only on EVERY trial and the
#    default sampler hits its actual shape 0.04% of the time. Reweighting the subset draw cannot
#    do this: write-only needs `chosen` inside a 2-element set out of 7, so all mass on k=1 still
#    caps it at 28.6%. Guarded by experiments/test_sampler_mix.py.
#    This is NOT a new sampler -- it is a mix change on version_B's existing one, and it is not a
#    re-run of version_C, which went to ~60% write-only and regressed. 0.35 is deliberately well
#    short of that.
#
# WHAT IS DELIBERATELY NOT HERE: a new attack sampler. Sampling has failed fixed (v8), widened
# (version_A/B) and adaptive-against-a-live-optimiser (version_C, which trained write-only ~60%
# of steps and regressed). Measured on version_B's sampler, write-only is 5.9% of steps and
# heretic's actual shape (write-only + near-full-stack) is 0.04% -- but version_C already
# bought that coverage and it did not place the entanglement. Structural, not a coverage gap.
#
# ALL MT-BENCH NUMBERS HERE ARE PINNED-JUDGE (deepseek-v4-flash-0731): base 4.54, gate-1 bar
# 4.04. Do NOT compare against the older floating-judge figures (base 4.96 / bar 4.46).
#
# THE HONEST PREDICTION, recorded before the run so the result is falsifiable either way:
# gate 1 PASSES (ART is 4.39 with clean-start 250; clean-start-0 was worth +1.4 on E1->E2) and
# gate 2 FAILS well above 0.3577, because both gate-2 ingredients failed gate 2 alone (ART
# 0.7058, Shairah 0.5365-0.7865) and gib 2 is half what version_B carried. Run it anyway
# because it answers two unanswered questions at once: "do ART and Shairah compose against
# heretic?" (nothing in the literature or this repo says) and "is the fluency/wall trade
# continuous or a switch?".
#
# ALREADY MEASURED -- do not re-litigate inside this run. Capability under heretic is 93-102%
# of own-clean for EVERY model and seed, so the poison pill never fires on a BENCHMARK under
# heretic; the only crater is version_B under rank-1 (GSM8K 0.0091). And gate 2 does not track
# generic competence, it tracks FLUENCY, which ARC/MMLU/GSM8K cannot see: heretic'd version_B
# holds 95% GSM8K at MT-Bench 2.96. Judge any gate-2 result against MT-Bench on the ATTACKED
# model, never against its benchmark scores.
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
    --version-a-p-canonical 0.10 --version-b-p-heretic 0.35 \
    --no-grad-checkpoint --recompute-direction-every 25 \
    --lambda-gib 2 --stage2-lambda-gib 2 \
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
