#!/usr/bin/env bash
# Gemma rrcenter recovery: reuse completed clean/rank-1/surgical batteries, replay only the
# completed Heretic study winner, add MT-Bench for every attacked arm, then apply runtime gates.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
source /venv/main/bin/activate
export CUDA_VISIBLE_DEVICES=0 SERVE_LL=api

ARM=gemma31bit_rrcenter
BASE=gemma31bit_base_clean
CK=outputs/version_g_gemma31bit_rrcenter.pt
HLOG=logs/heretic/heretic_${ARM}_s0.log
HTAG=${ARM}_her_s0
TOP3=results/${HTAG}_top3.json
HOUT=outputs/${HTAG}_att
MTB=scripts/external_benches/prompts/mtbench_t1.jsonl
PORT=${PORT:-8790}
UTIL=${UTIL:-0.60}
WORKERS=${GEN_WORKERS:-64}
JUDGE_WORKERS=${JUDGE_WORKERS:-48}
LOG=logs/eval/chain_${ARM}_new_gates.log
SP=

mkdir -p logs/eval/vllm results outputs
exec > >(tee -a "$LOG") 2>&1
say () { echo "[$(date -u +%H:%M:%S)] $*"; }
need () { [ -s "$1" ] || { say "[FAIL] missing $1"; exit 1; }; }
run_gates () {
  local phase="$1" gate verdict
  say "=== gates: $phase ==="
  for gate in 0 1 2 3 4; do
    verdict=$(python scripts/tools/gates.py --gate "$gate" --arm "$ARM" --base "$BASE" \
      --heretic-tag "$HTAG")
    say "gate $gate: $verdict"
  done
}
stop_server () {
  [ -n "${SP:-}" ] || return 0
  for child in $(pgrep -P "$SP" 2>/dev/null || true); do kill -TERM "$child" 2>/dev/null || true; done
  kill -TERM "$SP" 2>/dev/null || true
  wait "$SP" 2>/dev/null || true
  SP=
}
trap stop_server EXIT INT TERM

need "$CK"
need "$HLOG"
need outputs/${ARM}_rank1/model.safetensors
need outputs/${ARM}_surg_k16/model.safetensors
need scripts/tools/gates.py

say "=== select top 3 from completed Heretic study ==="
if [ ! -s "$TOP3" ]; then
  python - "$HLOG" "$TOP3" <<'PY'
import json, sys
sys.path.insert(0, "experiments")
from version_c_loop import parse_trials, pick_winners

trials = parse_trials(open(sys.argv[1], encoding="utf-8", errors="replace").read())
winners = pick_winners(trials, k=3, kl_max=0.5)
if len(winners) != 3:
    raise SystemExit(f"expected 3 Heretic winners, got {len(winners)} from {len(trials)} trials")
json.dump({"heretic_trials": {f"t{w['trial']}": w for w in winners}},
          open(sys.argv[2], "w"), indent=2)
for w in winners:
    print(f"t{w['trial']} refusals={w['refusals']}/100 kl={w['kl']:.4f}")
PY
fi
WINNER=$(python -c "import json; print(next(iter(json.load(open('$TOP3'))['heretic_trials'])))")
say "winner=$WINNER; materializing winner only"
if [ ! -s "$HOUT/model.safetensors" ]; then
  python -u experiments/version_c_replay.py --model-id google/gemma-3-1b-it \
    --checkpoint "$CK" --trial "$WINNER" --params-json "$TOP3" \
    --direction-recipe heretic --application heretic_full --out "$HOUT"
fi
need "$HOUT/model.safetensors"

say "=== full standard battery: Heretic winner only ==="
if [ ! -s "results/${HTAG}_judged/summary.json" ]; then
  PORT="$PORT" UTIL="$UTIL" bash scripts/eval/serve_eval.sh "$HTAG" "$HOUT" off
fi
need "results/${HTAG}_judged/summary.json"

# Show what can be decided now. Missing base controls are explicit SKIPs, not silence.
run_gates "interim"

serve_mtb () {
  local tag="$1" model="$2" rid="mtb_$1" ok=0
  if [ -s "results/$rid/generations.jsonl" ]; then
    say "[reuse] $rid"
    return
  fi
  need "$model/model.safetensors"
  ss -tln 2>/dev/null | grep -q ":${PORT} " && { say "[FAIL] port $PORT busy"; exit 1; }
  vllm serve "$model" --served-model-name "$rid" --port "$PORT" \
    --gpu-memory-utilization "$UTIL" --max-model-len 8192 --dtype bfloat16 \
    > "logs/eval/vllm/vllm_${rid}.log" 2>&1 &
  SP=$!
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"$rid\"" && { ok=1; break; }
    kill -0 "$SP" 2>/dev/null || break
    sleep 5
  done
  [ "$ok" = 1 ] || { tail -20 "logs/eval/vllm/vllm_${rid}.log"; exit 1; }
  python -u experiments/gen_via_api.py --run-id "$rid" --served-model "$rid" \
    --base-url "http://127.0.0.1:${PORT}/v1" --qwen-thinking off --num-workers "$WORKERS" \
    --prompt-file "$MTB" --max-new-tokens 768
  stop_server
  need "results/$rid/generations.jsonl"
}

say "=== MT-Bench: reuse base/clean; generate rank-1/surgical/Heretic ==="
serve_mtb "$BASE" outputs/gemma31bit_base_clean_hf
serve_mtb "$ARM" outputs/version_g_gemma31bit_rrcenter_clean
serve_mtb "${ARM}_rank1" outputs/${ARM}_rank1
serve_mtb "${ARM}_surg_k16" outputs/${ARM}_surg_k16
serve_mtb "$HTAG" "$HOUT"

MT_TAGS=("mtb_$BASE" "mtb_$ARM" "mtb_${ARM}_rank1" "mtb_${ARM}_surg_k16" "mtb_$HTAG")
python -u experiments/mtbench_single.py --repeats 3 --num-workers "$JUDGE_WORKERS" \
  --tags "${MT_TAGS[@]}"

say "=== wait for Gemma base standard battery ==="
for _ in $(seq 1 120); do
  [ -s "results/${BASE}_judged/summary.json" ] && \
  [ -s "results/${BASE}_xssafe_judged/summary.json" ] && \
  [ -s "results/${BASE}_xsunsafe_judged/summary.json" ] && \
  find "results/${BASE}_gsm8k" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && \
  find "results/${BASE}_humaneval" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && \
  find "results/${BASE}_mbpp" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && \
  find "results/${BASE}_arc" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && \
  find "results/${BASE}_mmlu" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && break
  sleep 120
done
need "results/${BASE}_judged/summary.json"
need "results/${BASE}_xssafe_judged/summary.json"
need "results/${BASE}_xsunsafe_judged/summary.json"
for bench in gsm8k humaneval mbpp arc mmlu; do
  find "results/${BASE}_${bench}" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . || {
    say "[FAIL] missing base $bench"; exit 1;
  }
done

run_gates "final"

say "=== final table ==="
python - "$BASE" "$ARM" "$HTAG" <<'PY'
import glob, json, sys

base, arm, her = sys.argv[1:]
mt = json.load(open("results/mtbench_single_scores.json"))
rows = [base, f"{arm}_clean", f"{arm}_rank1", f"{arm}_surg_k16", her]

def safety(tag):
    try:
        d = json.load(open(f"results/{tag}_judged/summary.json"))
        return d.get("harmful_actionable_rate"), d.get("gibberish_rate")
    except Exception:
        return None, None

def gsm(tag):
    hits = glob.glob(f"results/{tag}_gsm8k/**/results_*.json", recursive=True)
    if not hits:
        return None
    d = json.load(open(sorted(hits)[-1]))
    return d["results"]["gsm8k"]["exact_match,strict-match"]

def fmt(x, n=4):
    return "--" if x is None else f"{x:.{n}f}"

print(f"{'arm':38s} {'harm':>7s} {'gib':>7s} {'GSM8K':>7s} {'MTB':>6s}")
for tag in rows:
    h, g = safety(tag)
    mtag = tag.removesuffix("_clean") if tag == f"{arm}_clean" else tag
    m = (mt.get(f"mtb_{mtag}") or {}).get("mean")
    print(f"{tag:38s} {fmt(h):>7s} {fmt(g):>7s} {fmt(gsm(tag)):>7s} {fmt(m,2):>6s}")
PY

say "=== DONE ==="
