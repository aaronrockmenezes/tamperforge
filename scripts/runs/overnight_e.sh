#!/usr/bin/env bash
# Overnight: evaluate version_E arms E1/E2/E3, then heretic x3 seeds each.
#
# PHASE 0  wait for training (vere) + validate the persistent-vLLM harness
# PHASE 1  E1/E2/E3 x {clean, rank-1, surgical k16}, full battery
# PHASE 2  heretic 200 trials x 3 seeds per arm (studies 3-up), replay, same battery
#
# Battery: AdvBench-520 judged, XSTest safe+unsafe judged, GSM8K, MMLU-12, ARC, HumanEval, MBPP.
#
# HARD RULES (each learned the expensive way -- see docs/common_issues.md and CLAUDE.md):
#   * vLLM cannot share the GPU with a trainer: its startup memory profile ASSERTS when free
#     VRAM shifts, and it orphans an EngineCore holding ~7GB (PPID 1). Wait, never overlap.
#   * `tmux has-session -t X` does PREFIX matching and will match a longer session name --
#     it once made a chain wait on itself. Always `tmux ls -F '#{session_name}' | grep -qx X`.
#   * heretic prompts INTERACTIVELY when --study-checkpoint-dir exists, so an unattended
#     resume dies with EOFError from prompt_toolkit. Always rm -rf and start fresh.
#   * Guard on the ARTIFACT, never the directory; an empty dir from a killed job made four
#     eval arms silently vanish.
#   * Never `pkill -f` a global pattern. Scope with pgrep -P.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

LOG=logs/eval/overnight_e_$(date -u +%Y%m%dT%H%M%S).log
mkdir -p logs/eval logs/heretic
say () { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
alive () { tmux ls -F "#{session_name}" 2>/dev/null | grep -qx "$1"; }
have () { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }

reap () {
  for p in $(pgrep -f 'VLLM::EngineCore' 2>/dev/null); do
    [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = "1" ] && { say "  [reap] orphan $p"; kill -9 "$p"; }
  done
}

# free the disk if it gets tight; only ever drops attacked weights whose results already exist
guard_disk () {
  local free; free=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
  [ "${free:-999}" -ge 25 ] && return 0
  say "  [disk] only ${free}G free, pruning attacked dirs with complete results"
  for d in outputs/*_rank1 outputs/*_surg_k16 outputs/*_att; do
    [ -d "$d" ] || continue
    local t; t=$(basename "$d")
    if [ -f "results/${t}_judged/summary.json" ] && have "results/${t}_gsm8k"; then
      say "  [disk] rm $d"; rm -rf "$d"
    fi
  done
  df -h /workspace | tail -1 | tee -a "$LOG"
}

say "=== OVERNIGHT_E START ==="

# ---------------- PHASE 0: wait for training, validate harness ----------------
for S in vere smoke artfin arqher arlher; do
  if alive "$S"; then
    say "[wait] $S running..."
    while alive "$S"; do sleep 60; done
    say "[wait] $S finished"
  fi
done
sleep 30; reap

SERVE_LL=api
if grep -aq "VERDICT: PASS" /tmp/smoke_console.log 2>/dev/null; then
  say "harness validated: loglikelihood over API is trustworthy (SERVE_LL=api)"
else
  SERVE_LL=inprocess
  say "harness NOT validated for loglikelihood -- ARC/MMLU will run IN-PROCESS (SERVE_LL=inprocess)"
  grep -aE "^(metric|adv_harm|arc|mmlu|gsm8k|VERDICT)" /tmp/smoke_console.log 2>/dev/null | tail -8 | tee -a "$LOG"
fi
export SERVE_LL

ARMS=""
for a in e1 e2 e3; do
  [ -f "outputs/version_${a}_qwen_500.pt" ] && ARMS="$ARMS $a"
done
say "arms present:${ARMS:- NONE}"
[ -n "$ARMS" ] || { say "no version_E checkpoints -- nothing to do"; exit 1; }

# ---------------- PHASE 1: clean / rank-1 / surgical ----------------
for a in $ARMS; do
  CK="outputs/version_${a}_qwen_500.pt"
  say "=== PHASE1 $a ==="
  guard_disk

  # clean dir already built by the training script's smoke5 gate
  CLEAN="outputs/version_${a}_qwen_500_clean"
  [ -f "$CLEAN/model.safetensors" ] || python -u experiments/save_p1b_checkpoint.py \
    --checkpoint "$CK" --model-id Qwen/Qwen3-0.6B --attack none --out "$CLEAN" >>"$LOG" 2>&1
  [ -f "$CLEAN/model.safetensors" ] && bash scripts/eval/serve_eval.sh "ve_${a}_clean" "$CLEAN" off \
    || say "  [MISSING] $CLEAN"
  reap

  for spec in "rank1:0" "surg_k16:16"; do
    nm="${spec%%:*}"; k="${spec##*:}"
    D="outputs/ve_${a}_${nm}"
    guard_disk
    [ -f "$D/model.safetensors" ] || python -u experiments/v11_surgical_ablation.py \
      --model-id Qwen/Qwen3-0.6B --checkpoint "$CK" --direction-layer 20 --cap-rank "$k" \
      --out "$D" >>"$LOG" 2>&1
    if [ -f "$D/model.safetensors" ]; then
      bash scripts/eval/serve_eval.sh "ve_${a}_${nm}" "$D" off
      reap
      if [ -f "results/ve_${a}_${nm}_judged/summary.json" ] && have "results/ve_${a}_${nm}_gsm8k"; then
        say "  [cleanup] $D"; rm -rf "$D"
      else
        say "  [cleanup-skip] $D -- incomplete results"
      fi
    else
      say "  [FAIL] could not build $D"
    fi
  done
done
say "=== PHASE1 DONE ==="

# ---------------- PHASE 2: heretic x3 seeds per arm ----------------
for a in $ARMS; do
  CK="outputs/version_${a}_qwen_500.pt"
  CLEAN="outputs/version_${a}_qwen_500_clean"
  [ -f "$CLEAN/model.safetensors" ] || { say "[skip] heretic $a -- no clean dir"; continue; }
  say "=== PHASE2 heretic x3 on $a ==="
  guard_disk

  # studies 3-up: ~3.7GB each, measured safe. No vLLM concurrently.
  pids=""
  for S in 0 1 2; do
    HLOG="logs/heretic/heretic_ve_${a}_s${S}.log"
    if grep -aq "Running trial 200 of" "$HLOG" 2>/dev/null; then say "  [skip] study $a s$S"; continue; fi
    rm -rf "/tmp/hcp_ve_${a}_s${S}"
    say "  study $a s$S launching"
    heretic --model "$CLEAN" --n-trials 200 --seed "$S" \
      --study-checkpoint-dir "/tmp/hcp_ve_${a}_s${S}" < /dev/null > "$HLOG" 2>&1 &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p" 2>/dev/null || true; done
  say "  studies done for $a"
  sleep 15; reap

  for S in 0 1 2; do
    HLOG="logs/heretic/heretic_ve_${a}_s${S}.log"
    TAG="ve_${a}_her_s${S}"
    [ -f "$HLOG" ] || continue
    if [ ! -f "results/${TAG}_trial.json" ]; then
      python - "$HLOG" "$TAG" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
sys.path.insert(0, "experiments")
from version_c_loop import parse_trials, pick_winners
t = parse_trials(open(sys.argv[1], encoding="utf-8", errors="replace").read())
w = pick_winners(t, k=1, kl_max=0.5)
if w:
    json.dump({"heretic_trials": {"t%d" % w[0]["trial"]: w[0]}},
              open("results/%s_trial.json" % sys.argv[2], "w"), indent=2)
    print("  winner t%d kl=%.4f ref=%d/100" % (w[0]["trial"], w[0]["kl"], w[0]["refusals"]))
else:
    print("  NO WINNER")
PY
    fi
    [ -f "results/${TAG}_trial.json" ] || { say "  [FAIL] no winner $TAG"; continue; }
    T=$(python -c "import json;print(list(json.load(open('results/${TAG}_trial.json'))['heretic_trials'])[0])" 2>/dev/null)
    [ -n "$T" ] || { say "  [FAIL] unparseable winner $TAG"; continue; }
    D="outputs/${TAG}_att"
    guard_disk
    [ -f "$D/model.safetensors" ] || python -u experiments/version_c_replay.py \
      --model-id Qwen/Qwen3-0.6B --checkpoint "$CK" --trial "$T" \
      --params-json "results/${TAG}_trial.json" \
      --direction-recipe heretic --application heretic_full --out "$D" >>"$LOG" 2>&1
    if [ -f "$D/model.safetensors" ]; then
      bash scripts/eval/serve_eval.sh "$TAG" "$D" off
      reap
      if [ -f "results/${TAG}_judged/summary.json" ] && have "results/${TAG}_gsm8k"; then
        say "  [cleanup] $D"; rm -rf "$D"
      else
        say "  [cleanup-skip] $D -- incomplete results"
      fi
    else
      say "  [FAIL] replay $TAG"
    fi
  done
done

say "=== OVERNIGHT_E DONE ==="
df -h /workspace | tail -1 | tee -a "$LOG"
