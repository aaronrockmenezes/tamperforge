#!/usr/bin/env bash
# MT-Bench turn-1 across every variant we have, all judged pairwise against base Qwen.
#
# WHY: the whole battery is safety prompts or narrow capability. GSM8K/ARC/MMLU score an
# extracted answer or a loglikelihood, so none of them notice a model that answers "2+2=?"
# with "**2+2=4**" four times, drifts into Chinese, or repeats a clause eleven times -- all
# of which our arms actually do. 80 open-ended questions across 8 categories is the missing
# instrument.
#
# Attacked dirs were deleted by the eval chain's cleanup (by design -- they are reproducible
# from checkpoint + trial json), so rebuild the three E2 attack variants first.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a

say () { echo "[$(date -u +%H:%M:%S)] $*"; }
PD=scripts/external_benches/prompts/mtbench_t1.jsonl
[ -f "$PD" ] || { say "[FAIL] $PD missing"; exit 1; }

# ---- rebuild E2 attack variants ----
CK=outputs/version_e2_qwen_500.pt
for spec in "ve_e2_rank1:0" "ve_e2_surg_k16:16"; do
  nm="${spec%%:*}"; k="${spec##*:}"
  [ -f "outputs/$nm/model.safetensors" ] && { say "[skip] $nm exists"; continue; }
  say "rebuilding $nm"
  python -u experiments/v11_surgical_ablation.py --model-id Qwen/Qwen3-0.6B \
    --checkpoint "$CK" --direction-layer 20 --cap-rank "$k" --out "outputs/$nm" 2>&1 | tail -1
done
if [ ! -f outputs/ve_e2_her_s0_att/model.safetensors ] && [ -f results/ve_e2_her_s0_trial.json ]; then
  T=$(python -c "import json;print(list(json.load(open('results/ve_e2_her_s0_trial.json'))['heretic_trials'])[0])")
  say "rebuilding ve_e2_her_s0_att (trial $T)"
  python -u experiments/version_c_replay.py --model-id Qwen/Qwen3-0.6B --checkpoint "$CK" \
    --trial "$T" --params-json results/ve_e2_her_s0_trial.json \
    --direction-recipe heretic --application heretic_full --out outputs/ve_e2_her_s0_att 2>&1 | tail -1
fi

# ---- generate MT-Bench turn-1 for every variant ----
gen_for () {   # $1=tag $2=model-dir
  local tag="$1" md="$2"
  [ -f "outputs/$md/model.safetensors" ] || { say "[MISSING] outputs/$md"; return 0; }
  [ -f "results/mtb_${tag}/generations.jsonl" ] && { say "[skip] mtb_$tag"; return 0; }
  say "=== mtbench gen $tag ==="
  for p in $(pgrep -f "VLLM::EngineCore|vllm serve" 2>/dev/null); do
    [ "$(ps -o ppid= -p "$p" 2>/dev/null|tr -d ' ')" = "1" ] && kill -9 "$p" 2>/dev/null
  done
  if ss -tln 2>/dev/null | grep -q ":8765 "; then say "  [FAIL] port busy"; return 0; fi
  vllm serve "outputs/$md" --served-model-name "mtb_$tag" --port 8765 \
    --gpu-memory-utilization 0.85 --max-model-len 8192 --dtype bfloat16 \
    > "logs/eval/vllm_mtb_${tag}.log" 2>&1 &
  local SP=$!
  local ok=0
  for i in $(seq 1 90); do
    curl -sf http://127.0.0.1:8765/v1/models 2>/dev/null | grep -q "\"mtb_${tag}\"" && { ok=1; break; }
    kill -0 "$SP" 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" = "1" ]; then
    python -u experiments/gen_via_api.py --run-id "mtb_${tag}" --served-model "mtb_${tag}" \
      --base-url http://127.0.0.1:8765/v1 --prompt-file "$PD" --qwen-thinking off \
      --max-new-tokens 768 --num-workers 16 2>&1 | tail -2
  else
    say "  [FAIL] server never advertised mtb_${tag}"
  fi
  for c in $(pgrep -P "$SP" 2>/dev/null); do kill -TERM "$c" 2>/dev/null; done
  kill -TERM "$SP" 2>/dev/null; sleep 8; kill -9 "$SP" 2>/dev/null; sleep 5
}

gen_for xbase_clean   xbase_clean_hf
gen_for vb_clean      xvb_clean
gen_for e1_clean      version_e1_qwen_500_clean
gen_for e2_clean      version_e2_qwen_500_clean
gen_for e3_clean      version_e3_qwen_500_clean
gen_for e2_rank1      ve_e2_rank1
gen_for e2_surg       ve_e2_surg_k16
gen_for e2_heretic    ve_e2_her_s0_att
# 2026-08-03: the missing cell. version_B under heretic keeps 95% GSM8K / 98% MMLU / 99% ARC,
# but none of those grade response text, and harmful_actionable needs fluent prose where
# strict-match GSM8K does not. If this scores ~3.5 like vb_clean, version_B's 0.3212 is a real
# defence; if it scores ~2, the harm number is fluency-limited and the gate-2 pass is an artifact.
gen_for vb_heretic    heretic_vb_t99

# ---- pairwise vs base ----
say "=== pairwise judging (all vs base) ==="
for t in vb_clean e1_clean e2_clean e3_clean e2_rank1 e2_surg e2_heretic vb_heretic; do
  [ -f "results/mtb_${t}/generations.jsonl" ] || { say "[skip judge] $t"; continue; }
  python -u experiments/mtbench_pairwise.py --a "mtb_${t}" --b mtb_xbase_clean \
    --label-a "$t" --label-b base 2>&1 | grep -aE "^===|win-rate|^  (base|vb_|e1_|e2_|e3_|tie)" | head -8
done

# ---- absolute single-answer scores; pairwise is NOT a substitute (ART won 60.0% pairwise and
# still failed gate 1 on its absolute 4.325). Re-scores every tag, cheap and keeps one file.
say "=== absolute single-answer scores ==="
python -u experiments/mtbench_single.py --model deepseek/deepseek-v4-flash-0731 --tags \
  mtb_xbase_clean mtb_vb_clean mtb_e1_clean mtb_e2_clean mtb_e3_clean \
  mtb_e2_rank1 mtb_e2_surg mtb_e2_heretic mtb_art_clean mtb_vb_heretic 2>&1 | grep -avE "it/s\]|\r"

say "=== MTBENCH ALL DONE ==="
df -h /workspace | tail -1
