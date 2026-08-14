#!/usr/bin/env bash
# End-to-end Phase 1 on a 2-GPU box (2x RTX 3090, 24GB each): train -> gate chain -> probes.
#
#   DRY_RUN=1 bash scripts/runs/chain_2gpu.sh      # validate everything, launch nothing
#   bash scripts/runs/chain_2gpu.sh                # for real, inside tmux
#
# WHAT THIS IS NOT: an eval chain. scripts/runs/chain_f.sh already implements gates 0-3 and is
# fully parameterised; this drives it, it does not duplicate it.
#
# ARMS (default "rrcenter,rrplain"): each is pinned to its own GPU and its own vLLM port and
# runs train -> chain_f.sh independently, so the two proceed in parallel.
#   rrcenter  version_G gemma + --rr-center
#   rrplain   version_G gemma, identical but no centring -- the same-box control
# One flag apart, same seed and data, so a difference in outcome is attributable.
#
# MEMORY. gemma-3-1b --train-scope all is 698M trainable (77M attn + 621M MLP; the 302M embedding
# is frozen), and _reroute_loss holds the ablated model AND the frozen base per step:
#   model 2.0 + W0 1.4 + overrides/graph ~2.5 + grads 1.4 + AdamW fp32 5.6 + acts ~1.5 = ~14.4 GB
# Fits 24GB with room. So the training command is version_G's recipe VERBATIM except --rr-center
# -- no adamw8bit, no forced grad checkpointing. That matters: if anything else moves, a
# difference in outcome against the original gemma version_G run is uninterpretable.
# On a 12GB card this would OOM; there you would need --optim adamw8bit and to drop
# --no-grad-checkpoint (~9.5 GB), at the cost of exact comparability.
set -uo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
PY="${PY:-python}"
command -v /venv/main/bin/python >/dev/null 2>&1 && PY=/venv/main/bin/python

DRY_RUN="${DRY_RUN:-0}"
ARMS="${ARMS:-rrcenter,rrplain}"
MODEL="${MODEL:-google/gemma-3-1b-it}"
DL="${DL:-14}"                       # gemma direction layer; swept on BASE, L14 is 3rd of 14.
STEPS="${STEPS:-500}"
HARM=data/harm_targets_qwen.json
REF="${REF:-data/extended_refusals_advbench.json}"
BASE_TAG="${BASE_TAG:-gbase_clean}"  # gemma base. NEVER gate gemma against Qwen's MT-Bench.
BASE_HF="${BASE_HF:-outputs/gbase_clean_hf}"
mkdir -p logs/training_runs logs/eval outputs

say  () { echo "[$(date -u +%H:%M:%S)] $*"; }
run  () { if [ "$DRY_RUN" = 1 ]; then echo "    DRY: $*"; else eval "$@"; fi; }
fail () { echo "[PREFLIGHT FAIL] $*"; PF=1; }
PF=0

arm_gpu()  { case "$1" in rrcenter) echo 0;; rrplain) echo 1;; esac; }
arm_port() { case "$1" in rrcenter) echo 8765;; rrplain) echo 8775;; esac; }
arm_tag()  { echo "version_g_gemma_$1"; }
arm_short(){ case "$1" in rrcenter) echo vgc;; rrplain) echo vgp;; esac; }

# ======================================================================== PREFLIGHT
say "=== PREFLIGHT ==="
for a in ${ARMS//,/ }; do
  [ -n "$(arm_gpu "$a")" ] || fail "unknown arm '$a' (want rrcenter | rrplain)"
done
for f in experiments/train_tamper_resistant_v8.py experiments/save_p1b_checkpoint.py \
         scripts/runs/chain_f.sh scripts/eval/serve_eval.sh \
         scripts/probes/posthoc_lrr.py scripts/probes/gamma_surgical_amplification.py \
         "$HARM" "$REF" data/advbench_harmful_behaviors.csv; do
  [ -e "$f" ] || fail "missing $f"
done
command -v tmux >/dev/null 2>&1 || fail "tmux not installed (chain_f.sh waits on a tmux session)"

# chain_f.sh waits on a tmux session by name and only then reads the checkpoint. If PORT is
# still hardcoded there, two concurrent arms silently fight over one vLLM server.
grep -q 'PORT="${PORT:-8765}"' scripts/runs/chain_f.sh \
  || fail "chain_f.sh PORT is not overridable -- concurrent arms will collide on 8765"

# Exit code matters here: a missing bitsandbytes must FAIL preflight, not print a warning and
# sail through. The first version of this only printed, so a real box would have passed preflight
# and then crashed after model load, minutes in.
$PY - "$DRY_RUN" <<'PY' || PF=1
import importlib, sys
dry = sys.argv[1] == "1"
bad = []
try:
    import torch
except ImportError:
    sys.exit("  FATAL: no torch")
if not torch.cuda.is_available():
    (print("  [warn] CUDA unavailable (ok for DRY_RUN)") if dry else bad.append("CUDA unavailable"))
else:
    n = torch.cuda.device_count()
    print(f"  gpus: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"    gpu{i}: {p.name} {p.total_memory/2**30:.1f} GiB sm_{p.major}{p.minor}")
        if p.total_memory / 2**30 < 20:
            bad.append(f"gpu{i} has <20GiB; this recipe needs ~14.4GB. Add --optim adamw8bit "
                       f"and drop --no-grad-checkpoint (~9.5GB), losing exact comparability.")
    if n < 2:
        print("  [warn] <2 GPUs: arms will contend for VRAM. Prefer ARMS=rrcenter.")
for mod, why in (("vllm", "the gate chain cannot serve"),):
    ok = importlib.util.find_spec(mod) is not None
    print(f"  {mod}: {'ok' if ok else 'MISSING'}")
    if not ok and not dry:
        bad.append(f"{mod} missing -- {why}")
if bad:
    print("  FATAL: " + "; ".join(bad))
    sys.exit(1)
PY
[ -n "${OPENROUTER_API_KEY:-}" ] || fail "OPENROUTER_API_KEY unset -- gates 0/1/2 all judge"
[ -n "${HF_TOKEN:-}" ]           || fail "HF_TOKEN unset -- gemma is gated"
[ "$PF" = 0 ] || { say "preflight failed; fix the above"; exit 1; }
say "preflight OK"

# ======================================================================== BASE (shared, once)
# Gate 1 compares MT-Bench against the arm's OWN base. chain_f.sh defaulted to Qwen's, which
# silently gates a gemma arm against a different architecture -- guard on the artifact, not the dir.
say "=== BASE: $BASE_TAG ==="
if [ -s "${BASE_HF}/model.safetensors" ] || [ -s "${BASE_HF}/config.json" ]; then
  say "  present, skip"
else
  run "$PY -u experiments/save_p1b_checkpoint.py --model-id '$MODEL' --out '$BASE_HF' \
        --qwen-thinking off --attack none"
fi

# ======================================================================== TRAIN (parallel)
say "=== TRAIN ==="
for a in ${ARMS//,/ }; do
  G=$(arm_gpu "$a"); TAG=$(arm_tag "$a"); S=$(arm_short "$a")
  EXTRA=""; [ "$a" = rrcenter ] && EXTRA="--rr-center"
  LOG="logs/training_runs/${TAG}.log"
  if [ -s "outputs/${TAG}.pt" ]; then say "  SKIP $TAG (checkpoint exists)"; continue; fi
  say "  launch $TAG on gpu$G (tmux: $S)"
  # Written to a file, then `tmux ... bash FILE`. Passing a multi-line pipeline as a quoted
  # tmux argument through `eval` silently loses it -- verified: the session starts, the command
  # never runs, and the log is never created. A file has no quoting layer to get wrong, and it
  # is inspectable and rerunnable by hand.
  TRAIN="logs/training_runs/train_${S}.sh"
  cat > "$TRAIN" <<TRAINEOF
#!/usr/bin/env bash
set -uo pipefail
cd "\$(dirname "\$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export CUDA_VISIBLE_DEVICES=$G
$PY -u experiments/train_tamper_resistant_v8.py \\
  --model-id '$MODEL' --out 'outputs/${TAG}.pt' \\
  --train-scope all --abliterate-layers all --attack-ensemble \\
  --attack-profile version_b --attack-layers all --direction-layer $DL \\
  --version-a-p-canonical 0.10 --version-b-p-heretic 0.35 \\
  --no-grad-checkpoint --recompute-direction-every 25 \\
  --lambda-rr 4 $EXTRA --harm-targets '$HARM' --rr-layers last_half \\
  --lambda-gib 0 --stage2-lambda-gib 0 \\
  --lambda-uncensor 4 --uncensor-margin 4 \\
  --lambda-harm 4 --harm-margin 4 \\
  --lambda-safe 4 --stage2-lambda-safe 4 \\
  --lambda-reg 0.1 --lambda-clean 3 \\
  --clean-gen-prompts 2 --clean-gen-tokens 64 \\
  --clean-start-step 0 --clean-ramp-steps 100 \\
  --refusal-file '$REF' --refusal-max-len 384 \\
  --gsm8k-probe-n 8 --gsm8k-probe-max-new 256 \\
  --n-direction 256 --version-a-n-cap 256 \\
  --steps $STEPS --eval-every 50 --save-every 500 --lr 1e-5 --seed 42 \\
  --qwen-thinking off 2>&1 | tee '$LOG'
TRAINEOF
  chmod +x "$TRAIN"
  bash -n "$TRAIN" || { say "  [FAIL] generated $TRAIN is not valid bash"; exit 1; }
  say "    wrote $TRAIN"
  run "tmux new-session -d -s '$S' 'bash $TRAIN'"
done

# =============================================== GATES -> PROBES -> ARCHIVE (one session/arm)
# These MUST be sequenced, not fired in parallel with training. The first version launched the
# probes immediately, so they would have run against checkpoints that did not exist yet, and
# archived an empty directory. One tmux session per arm, running the stages in order, makes the
# ordering structural instead of a comment that says "run after".
#
# chain_f.sh itself polls for the TRAINING session ($S) to disappear before it reads the
# checkpoint, so launching this now is correct -- it blocks on its own.
A="../tamperforge-archive/phase1_$(date +%Y%m%d)"
say "=== GATES -> PROBES -> ARCHIVE ==="
for a in ${ARMS//,/ }; do
  G=$(arm_gpu "$a"); TAG=$(arm_tag "$a"); S=$(arm_short "$a"); P=$(arm_port "$a")
  say "  post-train chain for $TAG (gpu$G, port $P, tmux ch_$S)"
  # Written to a file rather than inlined into `tmux new-session "..."`. Quoting a multi-line
  # pipeline through tmux is a reliable way to ship a bug you cannot see; a file is also
  # inspectable and rerunnable by hand if a stage dies.
  STAGE="logs/eval/stage_${S}.sh"
  cat > "$STAGE" <<STAGEEOF
#!/usr/bin/env bash
set -uo pipefail
cd "\$(dirname "\$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
export CUDA_VISIBLE_DEVICES=$G

PORT=$P UTIL=0.60 TAG='$TAG' SHORT='$S' WAIT_ON='$S' \\
  BASE_TAG='$BASE_TAG' BASE_HF='$BASE_HF' MODEL_ID='$MODEL' DIRECTION_LAYER=$DL \\
  bash scripts/runs/chain_f.sh 2>&1 | tee 'logs/eval/chain_${S}.log'

# Probes run even if a gate rejected -- knowing WHY it failed is the point, and L_rr is the
# instrument that was missing the first time round. Guard on the ARTIFACT, never the directory.
if [ -s 'outputs/${TAG}.pt' ]; then
  $PY scripts/probes/posthoc_lrr.py --model-id '$MODEL' --checkpoint 'outputs/${TAG}.pt' \\
      --layer $DL --n-pairs 16 --tag '${S}'
  $PY scripts/probes/posthoc_lrr.py --model-id '$MODEL' --checkpoint 'outputs/${TAG}.pt' \\
      --layer $DL --n-pairs 16 --center --tag '${S}_centered'
  $PY scripts/probes/gamma_surgical_amplification.py --model-id '$MODEL' \\
      --checkpoint 'outputs/${TAG}.pt' --layer $DL --tag '${S}'
else
  echo '[probe] SKIP: outputs/${TAG}.pt absent -- training produced no checkpoint'
fi

# Archive last. The absence of these events.jsonl files for the original gemma run is the sole
# reason Phase 0a needed a post-hoc probe at all, and it cost a day.
mkdir -p '$A'
cp -r results/tamper_resistant_p1b_*/ '$A/' 2>/dev/null || true
cp results/posthoc_lrr.json results/gamma_surgical_amplification.json '$A/' 2>/dev/null || true
cp 'logs/training_runs/${TAG}.log' 'logs/eval/chain_${S}.log' '$A/' 2>/dev/null || true
echo "[done] $TAG"
STAGEEOF
  chmod +x "$STAGE"
  bash -n "$STAGE" || { say "  [FAIL] generated $STAGE is not valid bash"; exit 1; }
  say "    wrote $STAGE"
  run "tmux new-session -d -s 'ch_$S' 'bash $STAGE'"
done

cat <<EOF

=== reference values (results/posthoc_lrr.json) ===
                     uncentred            centred
  Qwen  version_G    0.9854 -> 0.2458     0.9617 -> 0.2072
  gemma version_G    0.9866 -> 0.9522     0.7529 -> 0.3324

gemma's UNCENTRED row is the instrument failing, not the mechanism: centred, the same checkpoint
shows 55.9% of range rerouted. Read both.

Watch:  tmux ls
        tail -f logs/training_runs/version_g_gemma_rrcenter.log
L_rr is now on the periodic step line -- read its trajectory, do not wait for the gates.

Report gibberish rate + benign usability beside EVERY harm number.
A low-harm model that is simply broken has fooled this project seven times.
EOF
