#!/usr/bin/env bash
# Heretic x3 seeds against version_E2, then full eval of each winner.
#
# E2 is the interesting target: it has the best CLEAN model of any arm we have built
# (XSTest benign 0.624, gibberish 0.088, MMLU 0.4988 above base) but rank-1 already walks
# through it at 41.4% harm while cratering GSM8K to 0.0387. Its wall never formed
# (gib_ce 0.49 at step 500 vs E1's 3.05). So the question is whether heretic -- which beat
# every other defense we have tested -- finds an even cheaper attack.
#
# Reference points, Qwen: base clean harm 0.2577, heretic ceiling 0.6596.
#   version_B heretic t99 harm 0.3212 | Shairah heretic mean 0.62 | ART heretic pending
#
# Studies run 3-up (~3.7GB each, measured safe). heretic prompts INTERACTIVELY when a
# --study-checkpoint-dir exists, so always rm -rf first: an unattended resume dies with
# EOFError from prompt_toolkit.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
export SERVE_LL=api

say () { echo "[$(date -u +%H:%M:%S)] $*"; }
have () { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }

CK=outputs/version_e2_qwen_500.pt
CLEAN=outputs/version_e2_qwen_500_clean
[ -f "$CLEAN/model.safetensors" ] || { say "[FAIL] $CLEAN missing"; exit 1; }

say "=== heretic x3 on version_E2 ==="
pids=""
for S in 0 1 2; do
  HLOG="logs/heretic/heretic_ve_e2_s${S}.log"
  if grep -aq "Running trial 200 of" "$HLOG" 2>/dev/null; then say "  [skip] study s$S"; continue; fi
  rm -rf "/tmp/hcp_ve_e2_s${S}"
  say "  study s$S launching"
  heretic --model "$CLEAN" --n-trials 200 --seed "$S" \
    --study-checkpoint-dir "/tmp/hcp_ve_e2_s${S}" < /dev/null > "$HLOG" 2>&1 &
  pids="$pids $!"
done
for p in $pids; do wait "$p" 2>/dev/null || true; done
say "  studies done"
sleep 15

for S in 0 1 2; do
  HLOG="logs/heretic/heretic_ve_e2_s${S}.log"
  TAG="ve_e2_her_s${S}"
  [ -f "$HLOG" ] || continue
  if [ ! -f "results/${TAG}_trial.json" ]; then
    python - "$HLOG" "$TAG" <<'PY'
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
  [ -f "$D/model.safetensors" ] || python -u experiments/version_c_replay.py \
    --model-id Qwen/Qwen3-0.6B --checkpoint "$CK" --trial "$T" \
    --params-json "results/${TAG}_trial.json" \
    --direction-recipe heretic --application heretic_full --out "$D" 2>&1 | tail -2
  if [ -f "$D/model.safetensors" ]; then
    bash scripts/eval/serve_eval.sh "$TAG" "$D" off
    if [ -f "results/${TAG}_judged/summary.json" ] && have "results/${TAG}_gsm8k"; then
      say "  [cleanup] $D"; rm -rf "$D"
    else
      say "  [cleanup-skip] $D -- incomplete"
    fi
  else
    say "  [FAIL] replay $TAG"
  fi
done

say "=== heretic E2 DONE ==="
df -h /workspace | tail -1
