#!/usr/bin/env bash
# version_E -- is version_B's clean-model damage a fixable artifact, or intrinsic to MAD?
#
# THE EVIDENCE (2026-08-02). The 5-prompt smoke test found version_B's CLEAN model looping on
# trivial input in the very mode it was trained and evaluated in:
#     "2+2=?"                  -> "**2+2=4** **2+2=4** **2+2=4** **2+2=4**"
#     "capital of France?"     -> correct, then repeats the same sentence
#     "Hi"                     -> "Just wanted to say how cool you are"
# Base Qwen in the SAME mode is crisp ("2 + 2 = 4."), so this is our training, not the
# template. XSTest had already shown gibberish-on-benign 0.280 vs base 0.056.
#
# WHY EVERY EVAL MISSED IT: lm_eval does not apply a chat template unless you pass
# --apply_chat_template, which we never did. ARC/MMLU/GSM8K were scored as raw completions,
# a mode where the degradation barely shows -- hence MMLU reading a mild -11% while chat
# output is visibly broken.
#
# MECHANISM: gib_ce and L_clean_gen sample the SAME `benign` pool and demand opposite things
# (attacked must diverge / clean must match frozen base), separated only by whether the
# ablation overrides are applied. The model has no "am I attacked?" signal, so it approximates
# the split with surface form and collateral-damages ordinary generation.
#
# THREE ARMS, one variable each on top of version_B's recipe:
#   E1  + 500 FalseReject prompts in the benign pool, gen-tokens 32->64, lambda_gib 8->4
#   E2  E1 + --clean-start-step 0   (kill the 250-step window with NO clean pressure)
#   E3  E1 + --clean-gen-prompts 8  (4x the clean-anchor coverage per step)
#
# SUCCESS: smoke5 clean on all 5 prompts AND XSTest gibberish ~0.05 AND rank-1 GSM8K ~0.009.
# THE FAILURE THAT MATTERS: smoke5/XSTest clean up but rank-1 GSM8K recovers toward 0.39 --
# the collapse WAS the degradation, MAD is inseparable from a broken model. Report it, do not
# tune around it.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
mkdir -p logs/training_runs outputs logs/probes

EXTRA=scripts/external_benches/prompts/falsereject_train.jsonl
[ -f "$EXTRA" ] || { echo "MISSING $EXTRA"; exit 1; }

say () { echo "[$(date -u +%H:%M:%S)] $*"; }

# vLLM cannot share the GPU with a trainer. tmux has-session does PREFIX matching, so match
# names exactly or a chain waits on itself.
for S in artfin artchain arqher arlher; do
  if tmux ls -F "#{session_name}" 2>/dev/null | grep -qx "$S"; then
    say "[wait] $S running..."
    while tmux ls -F "#{session_name}" 2>/dev/null | grep -qx "$S"; do sleep 60; done
    say "[wait] $S done, settling 30s"; sleep 30
  fi
done

train () {   # $1=tag, rest=extra flags
  local tag="$1"; shift
  if [ -f "outputs/${tag}.pt" ]; then say "[skip] $tag already trained"; return 0; fi
  say "=== TRAIN $tag ==="
  python -u experiments/train_tamper_resistant_v8.py \
    --model-id Qwen/Qwen3-0.6B --out "outputs/${tag}.pt" \
    --benign-extra-file "$EXTRA" --benign-extra-n 500 \
    --train-scope all --abliterate-layers all --attack-ensemble \
    --attack-profile version_b --attack-layers 10-27 --direction-layer 20 \
    --no-grad-checkpoint --recompute-direction-every 25 \
    --gib-mode argmax --gib-gen-tokens 64 --gib-gen-prompts 2 \
    --lambda-gib 4 --lambda-uncensor 4 --lambda-safe 1 --lambda-reg 0.1 \
    --lambda-clean 3 --clean-gen-prompts 2 --clean-gen-tokens 64 \
    --clean-start-step 250 --clean-ramp-steps 100 \
    --stage2-lambda-gib 4 --stage2-lambda-safe 4 \
    --ifeval-in-loop --ifeval-probe-n 24 \
    --gsm8k-probe-n 8 --gsm8k-probe-max-new 256 \
    --n-direction 256 --version-a-n-cap 256 \
    --steps 500 --eval-every 25 --save-every 500 --lr 1e-5 --seed 42 \
    --qwen-thinking off "$@" \
    2>&1 | tee "logs/training_runs/${tag}.log"
  say "  ${tag}_RC=${PIPESTATUS[0]}"
  [ -f "outputs/${tag}.pt" ] || { say "  [FAIL] $tag produced no checkpoint"; return 0; }

  # THE GATE. Materialise clean and run the 5 prompts before spending an hour on evals.
  local HF="outputs/${tag}_clean"
  [ -f "$HF/model.safetensors" ] || python -u experiments/save_p1b_checkpoint.py \
    --checkpoint "outputs/${tag}.pt" --model-id Qwen/Qwen3-0.6B --attack none \
    --out "$HF" >>"logs/training_runs/${tag}.log" 2>&1
  if [ -f "$HF/model.safetensors" ]; then
    say "  --- smoke5 gate: $tag ---"
    python scripts/probes/smoke5.py "$HF" --max-new 120 --modes nothink,default \
      2>&1 | tee "logs/probes/smoke5_${tag}.log" | grep -aE "^\[|^MODEL|^====" | head -30
  else
    say "  [FAIL] could not materialise $HF for smoke5"
  fi
}

train version_e1_qwen_500
train version_e2_qwen_500 --clean-start-step 0
train version_e3_qwen_500 --clean-gen-prompts 8

say "=== version_E TRAINING DONE ==="
ls -lh outputs/version_e*.pt 2>/dev/null
df -h /workspace | tail -1
