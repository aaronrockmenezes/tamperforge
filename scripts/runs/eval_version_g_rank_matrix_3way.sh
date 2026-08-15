#!/usr/bin/env bash
# Fresh trained-model layer selection + AdvBench-520 for rank-k and surgical rank-k.
# Three independent vLLM servers share one large GPU at 30% memory each.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate

PY="${PY:-python}"
[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the materialized trained clean model}"
SHORT="${SHORT:?set a unique result prefix}"
GPU="${GPU:-0}"
UTIL="${UTIL:-0.30}"
PORTS_CSV="${PORTS:-8765,8766,8767}"
ATTACK_RANKS="${ATTACK_RANKS:-1,2,4,8,16}"
MATRIX_VARIANTS="${MATRIX_VARIANTS:-}"
RANK_K_ESTIMATOR="${RANK_K_ESTIMATOR:-arditi_residual}"
SWEEP_N="${SWEEP_N:-16}"
SWEEP_MAX_NEW="${SWEEP_MAX_NEW:-1024}"
SWEEP_SNAPSHOT_DEVICE="${SWEEP_SNAPSHOT_DEVICE:-model}"
SWEEP_LAYERS="${SWEEP_LAYERS:-}"
GEN_WORKERS="${GEN_WORKERS:-64}"
JUDGE_WORKERS="${JUDGE_WORKERS:-64}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1024}"
JUDGE_TIMEOUT_SECONDS="${JUDGE_TIMEOUT_SECONDS:-10}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SAFETY_MAX_NEW_TOKENS="${SAFETY_MAX_NEW_TOKENS:-1024}"
LOG_DIR="logs/eval/${SHORT}/rank_matrix_3way"
SWEEP_DIR="${SWEEP_DIR:-results/dl_sweeps/${SHORT}_rank_matrix}"
SWEEP_GEN_DIR="${SWEEP_GEN_DIR:-$SWEEP_DIR/generations}"
SWEEP="$SWEEP_DIR/summary.json"
BASES="$SWEEP_DIR/selected_bases.pt"
TMP_ROOT=""
mkdir -p "$LOG_DIR/vllm" "$SWEEP_GEN_DIR" results
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
rows_n() { [ -s "$1" ] && [ "$(wc -l < "$1")" -eq "$2" ]; }

model_ready "$MODEL_DIR" || die "missing clean model weights: $MODEL_DIR"
for cmd in vllm curl ss setsid; do command -v "$cmd" >/dev/null || die "$cmd not installed"; done
case "$RANK_K_ESTIMATOR" in svd|arditi_residual|partitioned) ;; *) die "bad RANK_K_ESTIMATOR";; esac
IFS=',' read -r -a PORTS_ARR <<< "$PORTS_CSV"
[ "${#PORTS_ARR[@]}" -eq 3 ] || die "PORTS must contain exactly three ports"
IFS=',' read -r -a RANKS_ARR <<< "$ATTACK_RANKS"
[ "${#RANKS_ARR[@]}" -gt 0 ] || die "ATTACK_RANKS is empty"
for port in "${PORTS_ARR[@]}"; do
  ss -tln 2>/dev/null | grep -q ":${port} " && die "port $port is already occupied"
done
export CUDA_VISIBLE_DEVICES="$GPU"

expected_json=$($PY - "$ATTACK_RANKS" "$MATRIX_VARIANTS" <<'PY'
import json, sys
ranks = sorted({int(x) for x in sys.argv[1].split(",") if x})
all_variants = [*(f"rank{k}" for k in ranks), *(f"surg_rank{k}_cap16" for k in ranks)]
requested = [x for x in sys.argv[2].split(",") if x] or all_variants
unknown = sorted(set(requested) - set(all_variants))
if unknown: raise SystemExit(f"unsupported MATRIX_VARIANTS: {unknown}")
print(json.dumps(requested))
PY
)

if [ ! -s "$SWEEP" ]; then
  say "fresh layer sweep: ranks {$ATTACK_RANKS}, plain + surgical capK=16"
  variant_args=()
  layer_args=()
  [ -n "$MATRIX_VARIANTS" ] && variant_args=(--variants "$MATRIX_VARIANTS")
  [ -n "$SWEEP_LAYERS" ] && layer_args=(--layers "$SWEEP_LAYERS")
  "$PY" -u scripts/probes/adaptive_attack_sweep.py \
    --model-id "$MODEL_DIR" --attack-ranks "$ATTACK_RANKS" \
    "${variant_args[@]}" \
    "${layer_args[@]}" \
    --rank-estimator "$RANK_K_ESTIMATOR" --n-harmful "$SWEEP_N" \
    --snapshot-device "$SWEEP_SNAPSHOT_DEVICE" \
    --max-new-tokens "$SWEEP_MAX_NEW" --judge-workers "$JUDGE_WORKERS" \
    --judge-max-tokens "$JUDGE_MAX_TOKENS" \
    --judge-timeout-seconds "$JUDGE_TIMEOUT_SECONDS" \
    --out "$SWEEP" --gen-dir "$SWEEP_GEN_DIR"
fi
"$PY" - "$SWEEP" "$MODEL_DIR" "$expected_json" <<'PY' || die "invalid or stale sweep"
import json, pathlib, sys, torch
summary, model, expected = sys.argv[1], sys.argv[2], set(json.loads(sys.argv[3]))
d = json.load(open(summary))
assert d["model_id"] == model and d.get("checkpoint") is None
assert set(d["selected"]) == expected
basis_path = pathlib.Path(d["basis_file"])
assert basis_path.is_file()
b = torch.load(basis_path, map_location="cpu", weights_only=True)
assert b["_meta"]["model_id"] == model and b["_meta"]["checkpoint"] is None
assert set(b["bases"]) == expected
PY

TMP_ROOT=$(mktemp -d "/workspace/tamperforge_rank_matrix_${SHORT}.XXXXXX")
declare -a SERVER_PIDS=() SERVER_TAGS=() SERVER_DIRS=() SERVER_PORTS=() GEN_PIDS=() JUDGE_PIDS=()
stop_servers() {
  local pid
  for pid in "${SERVER_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]:-}"; do [ -n "$pid" ] && wait "$pid" 2>/dev/null || true; done
  SERVER_PIDS=(); SERVER_TAGS=(); SERVER_DIRS=(); SERVER_PORTS=()
}
cleanup() {
  stop_servers
  [ -n "$TMP_ROOT" ] && [ -d "$TMP_ROOT" ] && rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

variant_rank() { echo "$1" | sed -E 's/^surg_//; s/^rank([0-9]+).*/\1/'; }
variant_cap() { [[ "$1" == surg_* ]] && echo 16 || echo 0; }
tag_for() { echo "${SHORT}_$1"; }

materialize() {
  local variant="$1" rank cap layer out
  rank=$(variant_rank "$variant"); cap=$(variant_cap "$variant")
  layer=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"][sys.argv[2]]["layer"])' "$SWEEP" "$variant")
  out="$TMP_ROOT/$variant"
  say "$variant: materialize fresh L$layer atkK=$rank capK=$cap"
  "$PY" -u experiments/v11_surgical_ablation.py --model-id "$MODEL_DIR" \
    --direction-layer "$layer" --refusal-rank "$rank" --cap-rank "$cap" \
    --rank-estimator "$RANK_K_ESTIMATOR" --basis-file "$BASES" --basis-key "$variant" \
    --out "$out" >"$LOG_DIR/materialize_${variant}.log" 2>&1
  model_ready "$out" || die "$variant materialization failed"
}

start_server() {
  local variant="$1" model="$2" port="$3" tag pid
  tag=$(tag_for "$variant")
  say "$variant: start vLLM port=$port util=$UTIL"
  setsid vllm serve "$model" --served-model-name "$tag" --port "$port" \
    --gpu-memory-utilization "$UTIL" --max-model-len "$MAX_MODEL_LEN" --dtype bfloat16 \
    >"$LOG_DIR/vllm/${variant}.log" 2>&1 &
  pid=$!
  SERVER_PIDS+=("$pid"); SERVER_TAGS+=("$tag"); SERVER_DIRS+=("$model"); SERVER_PORTS+=("$port")
}

wait_servers() {
  local i port pid tag ready
  for i in "${!SERVER_PIDS[@]}"; do
    port="${SERVER_PORTS[$i]}"; pid="${SERVER_PIDS[$i]}"; tag="${SERVER_TAGS[$i]}"; ready=0
    for _ in $(seq 1 90); do
      if curl -sf "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q "\"${tag}\""; then ready=1; break; fi
      kill -0 "$pid" 2>/dev/null || { tail -30 "$LOG_DIR/vllm/${tag#${SHORT}_}.log"; die "$tag server died"; }
      sleep 5
    done
    [ "$ready" -eq 1 ] || die "$tag server not ready after 450s"
    say "${tag#${SHORT}_}: server ready"
  done
}

start_generation() {
  local variant="$1" port="$2" tag rid gen
  tag=$(tag_for "$variant"); rid="$tag"; gen="results/$rid/generations.jsonl"
  if rows_n "$gen" 520; then say "$variant: reuse 520 generations"; return; fi
  say "$variant: generate AdvBench-520"
  "$PY" -u experiments/gen_via_api.py --run-id "$rid" --served-model "$tag" \
    --base-url "http://127.0.0.1:${port}/v1" --qwen-thinking off \
    --num-workers "$GEN_WORKERS" --prompt-source advbench \
    --max-new-tokens "$SAFETY_MAX_NEW_TOKENS" >"$LOG_DIR/generate_${variant}.log" 2>&1 &
  GEN_PIDS+=("$!")
}

start_judge() {
  local variant="$1" tag gen summary
  tag=$(tag_for "$variant"); gen="results/$tag/generations.jsonl"; summary="results/${tag}_judged/summary.json"
  summary_n "$summary" 520 && { say "$variant: reuse judged n=520"; return; }
  rows_n "$gen" 520 || die "$variant generation is not 520 rows"
  say "$variant: async judge"
  "$PY" -u experiments/judge_generations.py --generations "$gen" \
    --run-id "${tag}_judged" --num-workers "$JUDGE_WORKERS" \
    --judge-max-tokens "$JUDGE_MAX_TOKENS" \
    --judge-timeout-seconds "$JUDGE_TIMEOUT_SECONDS" \
    >"$LOG_DIR/judge_${variant}.log" 2>&1 &
  JUDGE_PIDS+=("$!")
}

mapfile -t VARIANTS < <($PY - "$expected_json" <<'PY'
import json, sys
for name in json.loads(sys.argv[1]): print(name)
PY
)
pending=()
for variant in "${VARIANTS[@]}"; do
  summary_n "results/$(tag_for "$variant")_judged/summary.json" 520 || pending+=("$variant")
done
say "matrix: ${#VARIANTS[@]} arms; pending=${#pending[@]}; three concurrent servers"

for ((start=0; start<${#pending[@]}; start+=3)); do
  wave=("${pending[@]:start:3}")
  SERVER_PIDS=(); SERVER_TAGS=(); SERVER_DIRS=(); SERVER_PORTS=(); GEN_PIDS=()
  active_variants=(); active_ports=()
  for i in "${!wave[@]}"; do
    variant="${wave[$i]}"; tag=$(tag_for "$variant")
    if rows_n "results/$tag/generations.jsonl" 520; then continue; fi
    slot="${#active_variants[@]}"; port="${PORTS_ARR[$slot]}"
    materialize "$variant"
    out="$TMP_ROOT/$variant"
    start_server "$variant" "$out" "$port"
    active_variants+=("$variant"); active_ports+=("$port")
  done
  if [ "${#SERVER_PIDS[@]}" -gt 0 ]; then
    wait_servers
    for i in "${!active_variants[@]}"; do
      start_generation "${active_variants[$i]}" "${active_ports[$i]}"
    done
    failed=0
    for pid in "${GEN_PIDS[@]}"; do wait "$pid" || failed=1; done
    [ "$failed" -eq 0 ] || die "one or more generation jobs failed"
    stop_servers
  fi
  for variant in "${wave[@]}"; do start_judge "$variant"; done
  for dir in "$TMP_ROOT"/*; do [ -d "$dir" ] && rm -rf "$dir"; done
done

say "waiting for ${#JUDGE_PIDS[@]} asynchronous judges"
failed=0
for pid in "${JUDGE_PIDS[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || die "one or more judges failed"
for variant in "${VARIANTS[@]}"; do
  summary_n "results/$(tag_for "$variant")_judged/summary.json" 520 || die "$variant judged summary incomplete"
done

"$PY" - "$SHORT" "$expected_json" >"results/${SHORT}_rank_matrix_report.json" <<'PY'
import json, pathlib, sys
short, variants = sys.argv[1], json.loads(sys.argv[2])
rows = []
for variant in variants:
    summary = json.load(open(pathlib.Path("results") / f"{short}_{variant}_judged" / "summary.json"))
    rows.append({
        "variant": variant, "n": summary["n"],
        "harmful_actionable_rate": summary["harmful_actionable_rate"],
        "gibberish_rate": summary["gibberish_rate"],
        "judge_refusal_rate": summary["judge_refusal_rate"],
        "parse_failures": summary["parse_failures"],
    })
json.dump({"run": short, "rows": rows}, sys.stdout, indent=2)
PY
say "COMPLETE: all ${#VARIANTS[@]} AdvBench arms judged n=520"
