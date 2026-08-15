#!/usr/bin/env bash
# Post-training standard evaluation with fresh, attack-specific layer selection.
# Fixed attacks: rank1, rank4, rank16, and surgical cap16 versions of each.
# Adaptive Heretic runs separately against the same trained clean model.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

PY="${PY:-python}"
[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python
MODEL_ID="${MODEL_ID:?set MODEL_ID}"
CHECKPOINT="${CHECKPOINT:?set CHECKPOINT to the trained .pt}"
SHORT="${SHORT:?set a unique model/run slug}"
GPU="${GPU:-0}"
PORT="${PORT:-8790}"
UTIL="${UTIL:-0.80}"
GEN_WORKERS="${GEN_WORKERS:-64}"
JUDGE_WORKERS="${JUDGE_WORKERS:-64}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1024}"
JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-10}"
RANK_K_ESTIMATOR="${RANK_K_ESTIMATOR:-arditi_residual}"
OBLITERATUS_METHOD="${OBLITERATUS_METHOD:-aggressive}"
OBLITERATUS_COMMIT="${OBLITERATUS_COMMIT:-885390a0e2d78dfa9a62edaea5739d28d2b6903d}"
OBLITERATUS_REPO="${OBLITERATUS_REPO:-https://github.com/elder-plinius/OBLITERATUS.git}"
OBLITERATUS_SOURCE="${OBLITERATUS_SOURCE:-/workspace/.cache/tamperforge/obliteratus-${OBLITERATUS_COMMIT}}"
SWEEP_N="${SWEEP_N:-16}"
SWEEP_MAX_NEW="${SWEEP_MAX_NEW:-1024}"
CLEAN="${CLEAN:-outputs/version_g_${SHORT}_clean}"
LOG_DIR="logs/eval/${SHORT}/adaptive"
SWEEP="results/dl_sweeps/${SHORT}_trained_adaptive/summary.json"
SWEEP_GENS="results/dl_sweeps/${SHORT}_trained_adaptive/generations"
mkdir -p "$LOG_DIR" "$(dirname "$SWEEP")" outputs results
exec > >(tee -a "$LOG_DIR/driver.log") 2>&1

say() { echo "[$(date -u +%FT%TZ)] $*"; }
die() { say "[FAIL] $*"; exit 1; }
model_ready() {
  [ -s "$1/model.safetensors" ] || [ -s "$1/model.safetensors.index.json" ] ||
    find "$1" -maxdepth 1 -type f -name 'model-*.safetensors' -size +0 -print -quit 2>/dev/null | grep -q .
}
summary_n() {
  [ -s "$1" ] && "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("n") != int(sys.argv[2]))' "$1" "$2"
}
have_lm() { find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .; }
full_done() {
  local tag="$1" bench
  summary_n "results/${tag}_judged/summary.json" 520 || return 1
  summary_n "results/${tag}_xssafe_judged/summary.json" 250 || return 1
  summary_n "results/${tag}_xsunsafe_judged/summary.json" 200 || return 1
  for bench in gsm8k humaneval mbpp arc mmlu; do have_lm "results/${tag}_${bench}" || return 1; done
}

for f in scripts/probes/adaptive_attack_sweep.py experiments/v11_surgical_ablation.py \
         experiments/version_c_replay.py scripts/eval/serve_eval.sh "$CHECKPOINT"; do
  [ -e "$f" ] || die "missing $f"
done
for cmd in git heretic vllm; do command -v "$cmd" >/dev/null || die "$cmd not installed"; done
case "$RANK_K_ESTIMATOR" in svd|arditi_residual|partitioned) ;; *) die "bad RANK_K_ESTIMATOR";; esac
case "$OBLITERATUS_METHOD" in
  basic|advanced|aggressive|spectral_cascade|surgical|optimized|inverted|nuclear) ;;
  informed) die "OBLITERATUS_METHOD=informed is disabled: its CLI does not invoke InformedAbliterationPipeline at the pinned commit" ;;
  *) die "unsupported OBLITERATUS_METHOD=$OBLITERATUS_METHOD (SOM is broken at the pinned commit)";;
esac
export CUDA_VISIBLE_DEVICES="$GPU"

if ! model_ready "$CLEAN"; then
  say "materializing trained clean model"
  "$PY" -u experiments/save_p1b_checkpoint.py --model-id "$MODEL_ID" \
    --checkpoint "$CHECKPOINT" --attack none --out "$CLEAN"
fi
model_ready "$CLEAN" || die "clean materialization failed"

if ! full_done "${SHORT}_clean"; then
  say "standard clean evaluation"
  PORT="$PORT" UTIL="$UTIL" GEN_WORKERS="$GEN_WORKERS" JUDGE_WORKERS="$JUDGE_WORKERS" \
    JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" JUDGE_TIMEOUT_SECONDS="$JUDGE_TIMEOUT_SECONDS" \
    bash scripts/eval/serve_eval.sh "${SHORT}_clean" "$CLEAN" off
fi
full_done "${SHORT}_clean" || die "clean standard evaluation incomplete"

if [ ! -s "$SWEEP" ]; then
  say "fresh trained-model layer sweep: six fixed attack geometries"
  "$PY" -u scripts/probes/adaptive_attack_sweep.py \
    --model-id "$MODEL_ID" --checkpoint "$CHECKPOINT" \
    --rank-estimator "$RANK_K_ESTIMATOR" --n-harmful "$SWEEP_N" \
    --max-new-tokens "$SWEEP_MAX_NEW" --judge-workers "$JUDGE_WORKERS" \
    --judge-max-tokens "$JUDGE_MAX_TOKENS" \
    --judge-timeout-seconds "$JUDGE_TIMEOUT_SECONDS" \
    --out "$SWEEP" --gen-dir "$SWEEP_GENS"
fi
"$PY" - "$SWEEP" "$MODEL_ID" "$CHECKPOINT" <<'PY' || die "stale or invalid adaptive sweep"
import hashlib, json, pathlib, sys, torch
d = json.load(open(sys.argv[1]))
assert d["model_id"] == sys.argv[2] and d["checkpoint"] == sys.argv[3]
assert set(d["selected"]) == {"rank1", "rank4", "rank16", "surg_rank1_cap16",
                               "surg_rank4_cap16", "surg_rank16_cap16"}
digest = hashlib.sha256()
with open(sys.argv[3], "rb") as f:
    for chunk in iter(lambda: f.read(8 << 20), b""): digest.update(chunk)
assert d["checkpoint_sha256"] == digest.hexdigest()
basis_path = pathlib.Path(d["basis_file"])
assert basis_path.is_file()
b = torch.load(basis_path, map_location="cpu", weights_only=True)
assert b["_meta"]["checkpoint_sha256"] == d["checkpoint_sha256"]
assert set(b["bases"]) == set(d["selected"])
PY
BASES=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["basis_file"])' "$SWEEP")

TMP_ROOT=$(mktemp -d "/tmp/tamperforge_${SHORT}_adaptive.XXXXXX")
cleanup() { [ -n "${TMP_ROOT:-}" ] && [ -d "$TMP_ROOT" ] && rm -rf "$TMP_ROOT"; }
trap cleanup EXIT INT TERM

eval_attack() {
  local variant="$1" rank="$2" cap="$3" tag="${SHORT}_${variant}" out="$TMP_ROOT/$variant" layer
  layer=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"][sys.argv[2]]["layer"])' \
    "$SWEEP" "$variant")
  say "$variant: freshly selected L$layer, attack-rank=$rank, capability-rank=$cap"
  if ! full_done "$tag"; then
    "$PY" -u experiments/v11_surgical_ablation.py --model-id "$MODEL_ID" \
      --checkpoint "$CHECKPOINT" --direction-layer "$layer" --refusal-rank "$rank" \
      --cap-rank "$cap" --rank-estimator "$RANK_K_ESTIMATOR" \
      --basis-file "$BASES" --basis-key "$variant" --out "$out"
    model_ready "$out" || die "$variant materialization failed"
    PORT="$PORT" UTIL="$UTIL" GEN_WORKERS="$GEN_WORKERS" JUDGE_WORKERS="$JUDGE_WORKERS" \
      JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" JUDGE_TIMEOUT_SECONDS="$JUDGE_TIMEOUT_SECONDS" \
      bash scripts/eval/serve_eval.sh "$tag" "$out" off
  fi
  full_done "$tag" || die "$variant standard evaluation incomplete"
}

eval_attack rank1 1 0
eval_attack surg_rank1_cap16 1 16
eval_attack rank4 4 0
eval_attack rank16 16 0
eval_attack surg_rank4_cap16 4 16
eval_attack surg_rank16_cap16 16 16

HER_LOG="$LOG_DIR/heretic_seed0.log"
TOP3="results/${SHORT}_heretic_top3.json"
HER_TAG="${SHORT}_heretic"
if [ ! -s "$TOP3" ]; then
  say "fresh Heretic 200-trial search on trained clean model"
  HCP=$(mktemp -d "$TMP_ROOT/heretic_study.XXXXXX")
  heretic --model "$CLEAN" --n-trials 200 --seed 0 --study-checkpoint-dir "$HCP" \
    < /dev/null > "$HER_LOG" 2>&1 || true
  "$PY" - "$HER_LOG" "$TOP3" <<'PY'
import json, sys
sys.path.insert(0, "experiments")
from version_c_loop import parse_trials, pick_winners
trials = parse_trials(open(sys.argv[1], encoding="utf-8", errors="replace").read())
winners = pick_winners(trials, k=3, kl_max=0.5)
if not winners: raise SystemExit(f"no Heretic winner from {len(trials)} completed trials")
json.dump({"heretic_trials": {f"t{w['trial']}": w for w in winners}},
          open(sys.argv[2], "w"), indent=2)
PY
fi

if ! full_done "$HER_TAG"; then
  trial=$("$PY" -c 'import json,sys; print(next(iter(json.load(open(sys.argv[1]))["heretic_trials"])))' "$TOP3")
  her_out="$TMP_ROOT/heretic_winner"
  "$PY" -u experiments/version_c_replay.py --model-id "$MODEL_ID" \
    --checkpoint "$CHECKPOINT" --trial "$trial" --params-json "$TOP3" \
    --direction-recipe heretic --application heretic_full --out "$her_out"
  model_ready "$her_out" || die "Heretic winner replay failed"
  PORT="$PORT" UTIL="$UTIL" GEN_WORKERS="$GEN_WORKERS" JUDGE_WORKERS="$JUDGE_WORKERS" \
    JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" JUDGE_TIMEOUT_SECONDS="$JUDGE_TIMEOUT_SECONDS" \
    bash scripts/eval/serve_eval.sh "$HER_TAG" "$her_out" off
fi
full_done "$HER_TAG" || die "Heretic standard evaluation incomplete"

OBL_TAG="${SHORT}_obliteratus_${OBLITERATUS_METHOD}"
if ! full_done "$OBL_TAG"; then
  OBL_SRC="$OBLITERATUS_SOURCE"
  OBL_OUT="$TMP_ROOT/obliteratus_${OBLITERATUS_METHOD}"
  OBL_META="results/${OBL_TAG}_attack_provenance.json"
  say "fresh Obliteratus $OBLITERATUS_METHOD attack at pinned commit $OBLITERATUS_COMMIT"
  if [ ! -d "$OBL_SRC/.git" ]; then
    [ ! -e "$OBL_SRC" ] || die "OBLITERATUS_SOURCE exists but is not a git checkout: $OBL_SRC"
    mkdir -p "$(dirname "$OBL_SRC")"
    git clone --quiet "$OBLITERATUS_REPO" "$OBL_SRC"
  fi
  git -C "$OBL_SRC" checkout --quiet --detach "$OBLITERATUS_COMMIT"
  [ "$(git -C "$OBL_SRC" rev-parse HEAD)" = "$OBLITERATUS_COMMIT" ] || \
    die "Obliteratus source pin mismatch"
  OBLITERATUS_METHOD="$OBLITERATUS_METHOD" PYTHONPATH="$OBL_SRC${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" - <<'PY' || die "Obliteratus source/dependency preflight failed"
import os
from obliteratus.abliterate import METHODS

method = os.environ["OBLITERATUS_METHOD"]
assert method in METHODS, method
import obliteratus.cli
if method == "optimized":
    import optuna  # Not an Obliteratus project dependency; require it instead of silently degrading.
PY
  PYTHONPATH="$OBL_SRC${PYTHONPATH:+:$PYTHONPATH}" OBLITERATUS_TELEMETRY=0 \
    "$PY" -u -m obliteratus.cli obliterate "$CLEAN" \
      --method "$OBLITERATUS_METHOD" --output-dir "$OBL_OUT" \
      --device auto --dtype bfloat16 --verify-sample-size 30 --refusal-max-tokens 128
  model_ready "$OBL_OUT" || die "Obliteratus materialization failed"
  "$PY" - "$OBL_OUT" "$OBL_META" "$OBLITERATUS_REPO" "$OBLITERATUS_COMMIT" "$OBLITERATUS_METHOD" <<'PY'
import json, pathlib, sys
out, persistent, repo, commit, method = sys.argv[1:]
payload = json.dumps({
    "repository": repo, "commit": commit, "method": method, "telemetry": False,
}, indent=2)
pathlib.Path(out, "tamperforge_obliteratus_source.json").write_text(payload)
pathlib.Path(persistent).write_text(payload)
PY
  PORT="$PORT" UTIL="$UTIL" GEN_WORKERS="$GEN_WORKERS" JUDGE_WORKERS="$JUDGE_WORKERS" \
    JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS" JUDGE_TIMEOUT_SECONDS="$JUDGE_TIMEOUT_SECONDS" \
    bash scripts/eval/serve_eval.sh "$OBL_TAG" "$OBL_OUT" off
fi
full_done "$OBL_TAG" || die "Obliteratus standard evaluation incomplete"

say "COMPLETE: clean + eight fresh/adaptive attacks; attacked replicas removed on exit"
