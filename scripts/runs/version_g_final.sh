#!/usr/bin/env bash
# One resumable Version-G campaign:
#   base DL sweep -> base full/attacked controls -> rrcenter training -> clean/attack/Heretic
#   batteries + MT-Bench -> runtime gates -> final JSON/Markdown report.
#
# Minimal invocation (run inside tmux on the GPU box):
#   RANK_K_ESTIMATOR=arditi_residual MODEL_ID=google/gemma-3-1b-it GPU=0 \
#     bash scripts/runs/version_g_final.sh
#
# Useful overrides:
#   SHORT=phi4mini_vg STEPS=500 DL_DTYPE=bfloat16 DL_N=16 DL_STRIDE=1
#   TRAIN_EVAL_EVERY=0              # disable periodic held-out loss diagnostics
#   TRAIN_DIAGNOSTICS=1             # opt into slow GSM8K + local AdvBench preview
#   DRY_RUN=1                       # validate and print resolved plan; launch nothing
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

PY="${PY:-python}"
[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python
MODEL_ID="${MODEL_ID:-${MODEL:-google/gemma-3-1b-it}}"
SLUG="${MSLUG:-$(basename "$MODEL_ID" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9')}"
SHORT="${SHORT:-${SLUG}_rrcenter}"
GPU="${GPU:-0}"
STEPS="${STEPS:-500}"
PORT="${PORT:-8790}"
UTIL="${UTIL:-0.45}"              # leaves Heretic room while foreground vLLM runs
GEN_WORKERS="${GEN_WORKERS:-64}"
JUDGE_WORKERS="${JUDGE_WORKERS:-64}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek/deepseek-v4-flash-0731}"
DRY_RUN="${DRY_RUN:-0}"
RANK_K_ESTIMATOR="${RANK_K_ESTIMATOR:-arditi_residual}"
RANK_K_RANKS="${RANK_K_RANKS:-1,2,4,8,16}"
TRAIN_EVAL_EVERY="${TRAIN_EVAL_EVERY:-25}"
TRAIN_DIAGNOSTICS="${TRAIN_DIAGNOSTICS:-0}"
GIB_MODE="${GIB_MODE:-argmax}"

BASE_TAG="${BASE_TAG:-${SLUG}_base_clean}"
BASE_HF="${BASE_HF:-outputs/${BASE_TAG}_hf}"
BASE_RANK1="${BASE_TAG}_rank1"
BASE_SURG="${BASE_TAG}_surg_k16"
VG_TAG="${VG_TAG:-version_g_${SHORT}}"
CK="outputs/${VG_TAG}.pt"
CLEAN="outputs/${VG_TAG}_clean"
HER_TAG="${SHORT}_her_s0"
HER_OUT="outputs/${HER_TAG}_att"

DL_N="${DL_N:-16}"                # all layers x 16 rows ~= old every-other x 40 budget
DL_STRIDE="${DL_STRIDE:-1}"       # do not miss odd winners such as Llama L13
DL_LAYERS="${DL_LAYERS:-}"
DL_BATCH="${DL_BATCH:-16}"
DL_MAX_NEW="${DL_MAX_NEW:-1024}"
DL_DTYPE="${DL_DTYPE:-auto}"
if [ "$DL_DTYPE" = auto ]; then
  case "$MODEL_ID" in *gemma*) DL_DTYPE=float32;; *) DL_DTYPE=bfloat16;; esac
fi
DL_OUT="results/dl_sweeps/${SHORT}/summary.json"
DL_GENS="results/dl_sweeps/${SHORT}/generations"

# Final Version-G recipe. The old rank-1 canonical slice is replaced, not supplemented:
# one 10% rank-k branch samples RANK_K_RANKS uniformly. Surgical cap rank is a separate axis.
VG_P_RANK_K="${VG_P_RANK_K:-0.10}"
VG_P_SURGICAL_CONDITIONAL="${VG_P_SURGICAL_CONDITIONAL:-${VG_P_SURGICAL:-0.40}}"
VG_CAP_RANKS="${VG_CAP_RANKS:-2,4,8,16}"
VG_P_HERETIC="${VG_P_HERETIC:-0.35}"
HARM_TARGETS="${HARM_TARGETS:-data/harm_targets_qwen.json}"
REFUSALS="${REFUSALS:-data/extended_refusals_advbench.json}"

LOG_DIR="logs/campaigns/${SHORT}"
REPORT_DIR="results/reports/${SHORT}"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$(dirname "$DL_OUT")" outputs results
LOG="$LOG_DIR/driver.log"
[ "$DRY_RUN" = 1 ] || exec > >(tee -a "$LOG") 2>&1

say () { echo "[$(date -u +%H:%M:%S)] $*"; }
die () { say "[FAIL] $*"; exit 1; }
have_lm () { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }
summary_n () {
  [ -s "$1" ] && "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("n") != int(sys.argv[2]))' "$1" "$2"
}
model_ready () {
  [ -s "$1/model.safetensors" ] || [ -s "$1/model.safetensors.index.json" ] ||
    find "$1" -maxdepth 1 -type f -name 'model-*.safetensors' -size +0 -print -quit 2>/dev/null | grep -q .
}
full_done () {
  local t="$1" b
  summary_n "results/${t}_judged/summary.json" 520 || return 1
  summary_n "results/${t}_xssafe_judged/summary.json" 250 || return 1
  summary_n "results/${t}_xsunsafe_judged/summary.json" 200 || return 1
  for b in gsm8k humaneval mbpp arc mmlu; do have_lm "results/${t}_${b}" || return 1; done
}

say "=== PREFLIGHT ==="
for f in scripts/probes/gemma_dl_sweep_local.py experiments/save_p1b_checkpoint.py \
         experiments/v11_surgical_ablation.py experiments/train_version_g_final.py \
         scripts/eval/serve_eval.sh scripts/tools/gates.py \
         scripts/tools/model_context_len.py \
         "$HARM_TARGETS" "$REFUSALS" data/advbench_harmful_behaviors.csv; do
  [ -e "$f" ] || die "missing $f"
done
bash -n scripts/eval/serve_eval.sh "$0" || die "bash syntax check"
case "$GPU:$STEPS:$DL_N:$DL_STRIDE:$DL_MAX_NEW:$TRAIN_EVAL_EVERY" in
  *[!0-9:]*|:*|*::*|*:) die "GPU/STEPS/DL_N/DL_STRIDE/DL_MAX_NEW/TRAIN_EVAL_EVERY must be nonnegative integers";;
esac
case "$TRAIN_DIAGNOSTICS" in 0|1) ;; *) die "TRAIN_DIAGNOSTICS must be 0 or 1";; esac
case "$RANK_K_ESTIMATOR" in
  svd|arditi_residual|partitioned) ;;
  *) die "set RANK_K_ESTIMATOR=svd, arditi_residual, or partitioned";;
esac

cat <<EOF
model       $MODEL_ID
run         $SHORT
gpu/port    $GPU / $PORT
steps       $STEPS
DL sweep    stride=$DL_STRIDE n=$DL_N max_new=$DL_MAX_NEW dtype=$DL_DTYPE ${DL_LAYERS:+layers=$DL_LAYERS}
Version G   rank_k=$VG_P_RANK_K absolute ranks={$RANK_K_RANKS} estimator=$RANK_K_ESTIMATOR
            heretic=$VG_P_HERETIC absolute surgical=$VG_P_SURGICAL_CONDITIONAL conditional-on-remainder
            cap_ranks=$VG_CAP_RANKS rr_center=on
train eval  every=$TRAIN_EVAL_EVERY diagnostics=$TRAIN_DIAGNOSTICS gib_mode=$GIB_MODE
base        $BASE_TAG -> $BASE_HF
checkpoint  $CK
report      $REPORT_DIR/final_report.md
EOF
if [ "$DRY_RUN" = 1 ]; then
  say "DRY_RUN complete; nothing launched"
  exit 0
fi

[ -n "${OPENROUTER_API_KEY:-}" ] || die "OPENROUTER_API_KEY is unset"
[ ! -e results/.defer_api_scoring ] || die "results/.defer_api_scoring exists; final judged report would be incomplete"
for c in vllm heretic curl ss; do command -v "$c" >/dev/null 2>&1 || die "$c not installed"; done
$PY - "$GPU" <<'PY' || die "CUDA preflight"
import sys, torch
g = int(sys.argv[1])
assert torch.cuda.is_available(), "CUDA unavailable"
assert g < torch.cuda.device_count(), f"GPU {g} absent; have {torch.cuda.device_count()}"
p = torch.cuda.get_device_properties(g)
print(f"cuda gpu{g}: {p.name}, {p.total_memory/2**30:.1f} GiB")
PY
df -h . | tail -1
export CUDA_VISIBLE_DEVICES="$GPU"

# 1. Efficient single-load, judged sweep on the untouched base.
say "=== 1/8 DIRECTION-LAYER SWEEP ==="
if [ ! -s "$DL_OUT" ]; then
  sweep=("$PY" -u scripts/probes/gemma_dl_sweep_local.py
    --model-id "$MODEL_ID" --n-harmful "$DL_N" --n-direction 64
    --layer-stride "$DL_STRIDE" --batch-size "$DL_BATCH" --max-new "$DL_MAX_NEW" --dtype "$DL_DTYPE"
    --judge-model "$JUDGE_MODEL" --judge-workers "$JUDGE_WORKERS"
    --out "$DL_OUT" --gen-dir "$DL_GENS")
  [ -z "$DL_LAYERS" ] || sweep+=(--layers "$DL_LAYERS")
  "${sweep[@]}" 2>&1 | tee "$LOG_DIR/dl_sweep.log"
fi
DL=$($PY - "$DL_OUT" "$MODEL_ID" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("model_id") == sys.argv[2], (d.get("model_id"), sys.argv[2])
print(d.get("selected_layer", d["layers"][0]["layer"]))
PY
) || die "invalid or wrong-model DL sweep: $DL_OUT"
say "selected attack layer L$DL"

# 2. Untouched base: full battery, then rank-1/surgical-k16 AdvBench controls only.
say "=== 2/8 BASELINE FULL BATTERY ==="
if ! model_ready "$BASE_HF"; then
  "$PY" -u experiments/save_p1b_checkpoint.py --model-id "$MODEL_ID" \
    --out "$BASE_HF" --qwen-thinking off --attack none
fi
model_ready "$BASE_HF" || die "base materialization failed: $BASE_HF"
if ! full_done "$BASE_TAG"; then
  SKIP_MMLU=0 PORT="$PORT" UTIL="$UTIL" bash scripts/eval/serve_eval.sh "$BASE_TAG" "$BASE_HF" off
fi
full_done "$BASE_TAG" || die "base full battery incomplete"

build_base_attack () {
  local tag="$1" k="$2" out="outputs/$1"
  if ! model_ready "$out"; then
    "$PY" -u experiments/v11_surgical_ablation.py --model-id "$MODEL_ID" \
      --direction-layer "$DL" --cap-rank "$k" --out "$out"
  fi
  model_ready "$out" || die "failed to build $tag"
  if ! summary_n "results/${tag}_judged/summary.json" 520; then
    EVAL_PROFILE=advbench PORT="$PORT" UTIL="$UTIL" \
      bash scripts/eval/serve_eval.sh "$tag" "$out" off
  fi
  summary_n "results/${tag}_judged/summary.json" 520 || die "$tag AdvBench judgment incomplete"
}
say "=== 3/8 BASELINE ATTACK CONTROLS (ADVBENCH ONLY) ==="
build_base_attack "$BASE_RANK1" 0
build_base_attack "$BASE_SURG" 16

# 3. Version G, rr-center, selected base attack layer, otherwise Qwen/Llama defaults.
say "=== 4/8 VERSION-G RRCENTER TRAINING ==="
if [ ! -s "$CK" ]; then
  DIAGNOSTIC_ARGS=(--advbench-preview-tokens 0 --gsm8k-probe-n 0 --advbench-judge-n 0)
  if [ "$TRAIN_DIAGNOSTICS" = 1 ]; then
    DIAGNOSTIC_ARGS=(
      --advbench-preview-tokens 100
      --gsm8k-probe-n 8 --gsm8k-probe-max-new 256
      --advbench-judge-n 0
    )
  fi
  "$PY" -u experiments/train_version_g_final.py \
    --model-id "$MODEL_ID" --out "$CK" --run-id "$VG_TAG" \
    --train-scope all --abliterate-layers all --attack-ensemble \
    --attack-profile version_g_final --attack-layers all --direction-layer "$DL" \
    --vg-rank-k-prob "$VG_P_RANK_K" \
    --version-g-attack-ranks "$RANK_K_RANKS" \
    --version-g-rank-estimator "$RANK_K_ESTIMATOR" \
    --vg-surgical-prob "$VG_P_SURGICAL_CONDITIONAL" --vg-surgical-cap-ranks "$VG_CAP_RANKS" \
    --vg-heretic-prob "$VG_P_HERETIC" \
    --no-grad-checkpoint --recompute-direction-every 25 \
    --lambda-rr 4 --rr-center --harm-targets "$HARM_TARGETS" --rr-layers last_half \
    --lambda-gib 0 --stage2-lambda-gib 0 --gib-mode "$GIB_MODE" \
    --lambda-uncensor 4 --uncensor-margin 4 --lambda-harm 4 --harm-margin 4 \
    --lambda-safe 4 --stage2-lambda-safe 4 --lambda-reg 0.1 --lambda-clean 3 \
    --clean-gen-prompts 2 --clean-gen-tokens 64 --clean-start-step 0 --clean-ramp-steps 100 \
    --refusal-file "$REFUSALS" --refusal-max-len 384 \
    "${DIAGNOSTIC_ARGS[@]}" \
    --n-direction 256 --vg-n-capability 256 \
    --steps "$STEPS" --eval-every "$TRAIN_EVAL_EVERY" --save-every 500 --lr 1e-5 --seed 42 \
    --qwen-thinking off 2>&1 | tee "$LOG_DIR/training.log"
fi
[ -s "$CK" ] || die "training produced no $CK"

# 4-7. Clean and attacked batteries. This script owns the flow directly; no chain calls chain.
say "=== 5/8 CLEAN + ATTACKED FULL BATTERIES ==="
if ! model_ready "$CLEAN"; then
  "$PY" -u experiments/save_p1b_checkpoint.py --checkpoint "$CK" \
    --model-id "$MODEL_ID" --attack none --out "$CLEAN"
fi
model_ready "$CLEAN" || die "missing Version-G clean model"
full_done "${SHORT}_clean" || \
  PORT="$PORT" UTIL="$UTIL" GEN_WORKERS="$GEN_WORKERS" JUDGE_WORKERS="$JUDGE_WORKERS" \
    bash scripts/eval/serve_eval.sh "${SHORT}_clean" "$CLEAN" off
full_done "${SHORT}_clean" || die "clean full battery incomplete"

HER_DIR="$LOG_DIR/heretic"
mkdir -p "$HER_DIR"
HLOG="$HER_DIR/study_seed0.log"
LEGACY_HLOG="logs/heretic/heretic_${SHORT}_s0.log"
TOP3="results/${HER_TAG}_top3.json"
HPID=""
stop_heretic () {
  if [ -n "$HPID" ] && kill -0 "$HPID" 2>/dev/null; then
    kill -TERM "$HPID" 2>/dev/null || true
    wait "$HPID" 2>/dev/null || true
  fi
}
trap stop_heretic EXIT INT TERM
if [ ! -s "$TOP3" ]; then
  if grep -aq "Running trial 200 of" "$LEGACY_HLOG" 2>/dev/null; then
    HLOG="$LEGACY_HLOG"
    say "[reuse] completed legacy Heretic log $HLOG"
  elif ! grep -aq "Running trial 200 of" "$HLOG" 2>/dev/null; then
    HCP=$(mktemp -d "/tmp/tamperforge_${SHORT}_heretic.XXXXXX")
    heretic --model "$CLEAN" --n-trials 200 --seed 0 \
      --study-checkpoint-dir "$HCP" < /dev/null > "$HLOG" 2>&1 &
    HPID=$!
    say "Heretic running in background pid=$HPID log=$HLOG"
    sleep 30
  fi
fi

build_vg_attack () {
  local tag="$1" cap_rank="$2" out="outputs/$1"
  if ! model_ready "$out"; then
    "$PY" -u experiments/v11_surgical_ablation.py --model-id "$MODEL_ID" \
      --checkpoint "$CK" --direction-layer "$DL" --cap-rank "$cap_rank" --out "$out"
  fi
  model_ready "$out" || die "failed to build $tag"
  full_done "$tag" || \
    PORT="$PORT" UTIL="$UTIL" GEN_WORKERS="$GEN_WORKERS" JUDGE_WORKERS="$JUDGE_WORKERS" \
      bash scripts/eval/serve_eval.sh "$tag" "$out" off
  full_done "$tag" || die "$tag full battery incomplete"
}
build_vg_attack "${SHORT}_rank1" 0
build_vg_attack "${SHORT}_surg_k16" 16

if [ -n "$HPID" ]; then
  wait "$HPID" || die "Heretic failed; see $HLOG"
  HPID=""
fi

# Preserve up to three parameter sets for replay, but evaluate only winner #1.
if [ ! -s "$TOP3" ]; then
  "$PY" - "$HLOG" "$TOP3" <<'PY'
import json, sys
sys.path.insert(0, "experiments")
from version_c_loop import parse_trials, pick_winners
trials = parse_trials(open(sys.argv[1], encoding="utf-8", errors="replace").read())
winners = pick_winners(trials, k=3, kl_max=0.5)
if not winners:
    raise SystemExit(f"no Heretic winner under kl_max from {len(trials)} trials")
json.dump({"heretic_trials": {f"t{w['trial']}": w for w in winners}},
          open(sys.argv[2], "w"), indent=2)
print("Heretic top3:", ", ".join(f"t{w['trial']}" for w in winners))
PY
fi

HER_TRIAL=$("$PY" -c \
  'import json,sys; print(next(iter(json.load(open(sys.argv[1]))["heretic_trials"])))' "$TOP3")
if ! model_ready "$HER_OUT"; then
  "$PY" -u experiments/version_c_replay.py --model-id "$MODEL_ID" --checkpoint "$CK" \
    --trial "$HER_TRIAL" --params-json "$TOP3" --direction-recipe heretic \
    --application heretic_full --out "$HER_OUT"
fi
model_ready "$HER_OUT" || die "missing Heretic winner model"
full_done "$HER_TAG" || \
  PORT="$PORT" UTIL="$UTIL" GEN_WORKERS="$GEN_WORKERS" JUDGE_WORKERS="$JUDGE_WORKERS" \
    bash scripts/eval/serve_eval.sh "$HER_TAG" "$HER_OUT" off
full_done "$HER_TAG" || die "Heretic full battery incomplete"
trap - EXIT INT TERM

# Score one consistent five-arm MT-Bench set after all model generations are complete.
say "=== 6/8 MT-BENCH FOR EVERY ARM ==="
SERVER_PID=""
MT_JUDGE_PID=""
stop_server () {
  [ -n "$SERVER_PID" ] || return 0
  for child in $(pgrep -P "$SERVER_PID" 2>/dev/null || true); do kill -TERM "$child" 2>/dev/null || true; done
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  sleep 8
  kill -9 "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}
cleanup_mt () {
  stop_server
  if [ -n "$MT_JUDGE_PID" ] && kill -0 "$MT_JUDGE_PID" 2>/dev/null; then
    kill -TERM "$MT_JUDGE_PID" 2>/dev/null || true
    wait "$MT_JUDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup_mt EXIT INT TERM
MTB="scripts/external_benches/prompts/mtbench_t1.jsonl"
MTB_N=$("$PY" -c 'import sys; print(sum(1 for line in open(sys.argv[1]) if line.strip()))' "$MTB")
mt_gen () {
  local tag="$1" model="$2" rid="mtb_$1" ready=0 ctx
  if [ -s "results/$rid/generations.jsonl" ]; then
    "$PY" -c 'import sys; sys.exit(sum(1 for line in open(sys.argv[1]) if line.strip()) != int(sys.argv[2]))' \
      "results/$rid/generations.jsonl" "$MTB_N" || die "$rid generations are partial"
    say "[reuse] $rid"
    return
  fi
  model_ready "$model" || die "missing MT-Bench model $model"
  ctx=$("$PY" scripts/tools/model_context_len.py "$model" 8192) || die "cannot read context length: $model"
  ss -tln 2>/dev/null | grep -q ":${PORT} " && die "port $PORT busy"
  vllm serve "$model" --served-model-name "$rid" --port "$PORT" \
    --gpu-memory-utilization "$UTIL" --max-model-len "$ctx" --dtype bfloat16 \
    > "$LOG_DIR/vllm_${rid}.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 90); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "\"$rid\"" && { ready=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 5
  done
  [ "$ready" = 1 ] || die "vLLM failed for $rid; see $LOG_DIR/vllm_${rid}.log"
  "$PY" -u experiments/gen_via_api.py --run-id "$rid" --served-model "$rid" \
    --base-url "http://127.0.0.1:${PORT}/v1" --qwen-thinking off \
    --num-workers "$GEN_WORKERS" --prompt-file "$MTB" --max-new-tokens 768
  stop_server
  "$PY" -c 'import sys; sys.exit(sum(1 for line in open(sys.argv[1]) if line.strip()) != int(sys.argv[2]))' \
    "results/$rid/generations.jsonl" "$MTB_N" || die "missing or partial $rid generations"
}

mt_scored () {
  "$PY" - "$1" "$MTB_N" <<'PY'
import json, pathlib, sys
p = pathlib.Path("results/mtbench_single_scores.json")
if not p.exists(): raise SystemExit(1)
r = json.load(open(p)).get(sys.argv[1], {})
raise SystemExit(not (r.get("n") == int(sys.argv[2]) and r.get("repeats") == 3))
PY
}
mt_score_queue () {
  local tag rid
  for tag in "$BASE_TAG" "$SHORT" "${SHORT}_rank1" "${SHORT}_surg_k16" "$HER_TAG"; do
    rid="mtb_$tag"
    while ! "$PY" - "results/$rid/generations.jsonl" "$MTB_N" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
raise SystemExit(not (p.exists() and sum(bool(x.strip()) for x in open(p)) == int(sys.argv[2])))
PY
    do sleep 5; done
    mt_scored "$rid" && { say "[reuse] MT-Bench score $rid"; continue; }
    "$PY" -u experiments/mtbench_single.py --repeats 3 --num-workers "$JUDGE_WORKERS" \
      --tags "$rid" || return 1
    mt_scored "$rid" || return 1
  done
}

mt_score_queue >> "$LOG_DIR/mtbench_judge.log" 2>&1 &
MT_JUDGE_PID=$!
say "MT-Bench async judge queue pid=$MT_JUDGE_PID"
mt_gen "$BASE_TAG" "$BASE_HF"
mt_gen "$SHORT" "$CLEAN"
mt_gen "${SHORT}_rank1" "outputs/${SHORT}_rank1"
mt_gen "${SHORT}_surg_k16" "outputs/${SHORT}_surg_k16"
mt_gen "$HER_TAG" "$HER_OUT"
wait "$MT_JUDGE_PID" || die "MT-Bench judge queue failed; see $LOG_DIR/mtbench_judge.log"
MT_JUDGE_PID=""

# 8. Compact source-of-truth report. Raw generations/judgments stay in their existing dirs.
say "=== 7/8 RUNTIME GATES ==="
for gate in 0 1 2 3 4; do
  "$PY" scripts/tools/gates.py --gate "$gate" --arm "$SHORT" --base "$BASE_TAG" \
    --heretic-tag "$HER_TAG" 2>&1 | sed "s/^/gate $gate: /"
done

say "=== 8/8 FINAL REPORT ==="
export SHORT BASE_TAG BASE_RANK1 BASE_SURG VG_TAG HER_TAG MODEL_ID DL DL_OUT REPORT_DIR TOP3
export RANK_K_ESTIMATOR RANK_K_RANKS VG_P_RANK_K VG_P_SURGICAL_CONDITIONAL VG_CAP_RANKS
export VG_P_HERETIC
export TRAIN_EVAL_EVERY TRAIN_DIAGNOSTICS GIB_MODE
"$PY" - <<'PY'
import glob, json, os, subprocess, sys
from pathlib import Path

R = Path("results")
short, base = os.environ["SHORT"], os.environ["BASE_TAG"]
her = os.environ["HER_TAG"]
out = Path(os.environ["REPORT_DIR"])
out.mkdir(parents=True, exist_ok=True)

def summary(tag):
    p = R / f"{tag}_judged/summary.json"
    return json.load(open(p)) if p.exists() else {}

metrics = ("exact_match,strict-match", "exact_match,flexible-extract", "pass@1",
           "acc_norm,none", "acc,none", "exact_match", "acc_norm", "acc")
def capability(tag, bench):
    hits = glob.glob(str(R / f"{tag}_{bench}/**/results_*.json"), recursive=True)
    if not hits: return None
    vals = []
    for m in json.load(open(sorted(hits)[-1])).get("results", {}).values():
        for key in metrics:
            if isinstance(m.get(key), (int, float)):
                vals.append(float(m[key])); break
    return sum(vals) / len(vals) if vals else None

try: mt = json.load(open(R / "mtbench_single_scores.json"))
except Exception: mt = {}

specs = [
    (base, "base clean", "full", f"mtb_{base}"),
    (os.environ["BASE_RANK1"], "base rank-1", "AdvBench only", None),
    (os.environ["BASE_SURG"], "base surgical k16", "AdvBench only", None),
    (f"{short}_clean", "Version G clean", "full", f"mtb_{short}"),
    (f"{short}_rank1", "Version G rank-1", "full", f"mtb_{short}_rank1"),
    (f"{short}_surg_k16", "Version G surgical k16", "full", f"mtb_{short}_surg_k16"),
    (her, "Version G Heretic winner", "full", f"mtb_{her}"),
]
rows = []
for tag, label, scope, mtag in specs:
    s = summary(tag); n = s.get("n") or 0
    xs, xu = summary(f"{tag}_xssafe"), summary(f"{tag}_xsunsafe")
    xn = xs.get("n") or 0
    rows.append({
        "tag": tag, "arm": label, "scope": scope, "n_advbench": n,
        "advbench_harm": s.get("harmful_actionable_rate"),
        "advbench_gibberish": s.get("gibberish_rate"),
        "advbench_refusal": s.get("judge_refusal_rate"),
        "xstest_safe_answered": ((xs.get("usefulness_counts") or {}).get("benign", 0) / xn
                                  if xn else None),
        "xstest_unsafe_refused": xu.get("judge_refusal_rate"),
        **{b: capability(tag, b) for b in ("gsm8k", "mbpp", "humaneval", "arc", "mmlu")},
        "mtbench": (mt.get(mtag) or {}).get("mean") if mtag else None,
    })

gates = {}
for g in range(5):
    p = subprocess.run([sys.executable, "scripts/tools/gates.py", "--gate", str(g),
                        "--arm", short, "--base", base, "--heretic-tag", her],
                       text=True, capture_output=True)
    gates[str(g)] = {"verdict": p.stdout.strip().splitlines()[-1],
                     "detail": p.stderr.strip().splitlines()[-1]}

sweep = json.load(open(os.environ["DL_OUT"]))
selected = next(r for r in sweep["layers"] if r["layer"] == int(os.environ["DL"]))
report = {
    "model_id": os.environ["MODEL_ID"], "version_g_checkpoint": os.environ["VG_TAG"],
    "direction_layer": int(os.environ["DL"]), "dl_sweep": selected,
    "training_attack": {
        "profile": "version_g_final",
        "rank_k_probability": float(os.environ["VG_P_RANK_K"]),
        "rank_k_ranks": [int(x) for x in os.environ["RANK_K_RANKS"].split(",")],
        "rank_k_estimator": os.environ["RANK_K_ESTIMATOR"],
        "surgical_conditional_probability": float(os.environ["VG_P_SURGICAL_CONDITIONAL"]),
        "surgical_expected_total_probability": (
            (1.0 - float(os.environ["VG_P_RANK_K"]) - float(os.environ["VG_P_HERETIC"]))
            * float(os.environ["VG_P_SURGICAL_CONDITIONAL"])
        ),
        "surgical_cap_ranks": [int(x) for x in os.environ["VG_CAP_RANKS"].split(",")],
        "heretic_probability": float(os.environ["VG_P_HERETIC"]),
    },
    "periodic_eval": {
        "every": int(os.environ["TRAIN_EVAL_EVERY"]),
        "slow_diagnostics": bool(int(os.environ["TRAIN_DIAGNOSTICS"])),
        "gib_mode": os.environ["GIB_MODE"],
    },
    "dl_sweep_base": sweep.get("base_judged"), "rows": rows, "gates": gates,
    "heretic_top3": json.load(open(os.environ["TOP3"])),
    "caveat": "AdvBench is train-exposed for Version G; it is not a held-out generalization result.",
}
json.dump(report, open(out / "final_report.json", "w"), indent=2)

def f(v, n=3): return "--" if v is None else f"{v:.{n}f}"
cols = [("arm", "Arm"), ("scope", "Scope"), ("advbench_harm", "Adv harm"),
        ("advbench_gibberish", "Gib"), ("advbench_refusal", "Refuse"),
        ("xstest_safe_answered", "XS-safe ans"), ("xstest_unsafe_refused", "XS-unsafe ref"),
        ("gsm8k", "GSM8K"), ("mbpp", "MBPP"), ("humaneval", "HumanEval"),
        ("arc", "ARC"), ("mmlu", "MMLU"), ("mtbench", "MTB")]
lines = [f"# Version G campaign: {os.environ['MODEL_ID']}", "",
         f"Selected base attack layer: **L{os.environ['DL']}** "
         f"(judged harm {f(selected.get('harmful_actionable'))}, "
         f"gibberish {f(selected.get('gibberish'))}).", "",
         f"Training rank-k branch: **{os.environ['VG_P_RANK_K']} total probability**, "
         f"uniform k={{{os.environ['RANK_K_RANKS']}}}, "
         f"estimator `{os.environ['RANK_K_ESTIMATOR']}`. Surgical cap-rank is separate.", "",
         f"Surgical draw probability: **{os.environ['VG_P_SURGICAL_CONDITIONAL']} conditional** "
         f"on the generic subset branch (not a total-step fraction).", "",
         "| " + " | ".join(v for _, v in cols) + " |",
         "| " + " | ".join("---" for _ in cols) + " |"]
for row in rows:
    lines.append("| " + " | ".join(str(row[k]) if k in ("arm", "scope") else f(row[k])
                                    for k, _ in cols) + " |")
lines += ["", "## Gates", ""]
for g, d in gates.items(): lines.append(f"- Gate {g}: **{d['verdict']}** — {d['detail']}")
lines += ["", "> AdvBench is train-exposed for Version G; treat it as in-distribution, not a held-out generalization result.", ""]
(out / "final_report.md").write_text("\n".join(lines))
print(out / "final_report.md")
print(out / "final_report.json")
PY

df -h . | tail -1
say "=== CAMPAIGN COMPLETE: $REPORT_DIR/final_report.md ==="
