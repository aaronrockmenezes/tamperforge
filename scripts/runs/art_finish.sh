#!/usr/bin/env bash
# Finish the ART heretic arm: wait for both seed-0 studies (running in parallel in their own
# tmux sessions), parse winners, replay, then evaluate via the PERSISTENT vLLM server.
#
# Studies run 2-up on purpose (~3.7GB each) -- the chain only parallelised across seeds, so
# at 1 seed it degenerated to sequential and wasted ~40min.
#
# Heretic prompts INTERACTIVELY when a --study-checkpoint-dir already exists, so a resume
# with stdin=/dev/null dies with EOFError from prompt_toolkit. Always rm -rf and start fresh;
# there is no unattended resume.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

say () { echo "[$(date -u +%H:%M:%S)] $*"; }

for S in arqher arlher; do
  if tmux ls -F "#{session_name}" 2>/dev/null | grep -qx "$S"; then
    say "[wait] study $S..."
    while tmux ls -F "#{session_name}" 2>/dev/null | grep -qx "$S"; do sleep 60; done
    say "[wait] $S done"
  fi
done
sleep 20

# arch | prefix | model-id | checkpoint | thinking
for spec in \
  "arq|Qwen/Qwen3-0.6B|outputs/art_qwen_500.pt|off" \
  "arl|meta-llama/Llama-3.2-1B-Instruct|outputs/art_llama_500.pt|default"
do
  IFS='|' read -r P MID CK TH <<<"$spec"
  HLOG="logs/heretic/heretic_${P}_s0.log"
  TAG="${P}_her_s0"
  n=$(grep -ac "Running trial" "$HLOG" 2>/dev/null || echo 0)
  say "=== $TAG (study trials seen: $n) ==="

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
  [ -f "results/${TAG}_trial.json" ] || { say "  [FAIL] no winner for $TAG"; continue; }

  T=$(python -c "import json;print(list(json.load(open('results/${TAG}_trial.json'))['heretic_trials'])[0])" 2>/dev/null)
  D="outputs/${TAG}_att"
  if [ ! -f "$D/model.safetensors" ]; then
    say "  replaying $T -> $D"
    python -u experiments/version_c_replay.py --model-id "$MID" --checkpoint "$CK" \
      --trial "$T" --params-json "results/${TAG}_trial.json" \
      --direction-recipe heretic --application heretic_full --out "$D" 2>&1 | tail -2
  fi
  [ -f "$D/model.safetensors" ] || { say "  [FAIL] replay $TAG"; continue; }

  say "  eval via persistent vLLM server"
  bash scripts/eval/serve_eval.sh "$TAG" "$D" "$TH"

  if [ -f "results/${TAG}_judged/summary.json" ] && \
     find "results/${TAG}_gsm8k" -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; then
    say "  [cleanup] $D"; rm -rf "$D"
  else
    say "  [cleanup-skip] $D -- incomplete results, keeping weights"
  fi
done

say "=== ART heretic DONE ==="
df -h /workspace | tail -1
