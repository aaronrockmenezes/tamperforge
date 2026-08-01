#!/usr/bin/env bash
# Does MAD key on the READ projections?
#
# The DL sweep settled that heretic's write-only shape never triggers the collapse at ANY
# direction layer (GSM8K 0.376-0.397 across DL 8..26, clean 0.3844), while rank-1 -- which
# hits read+write -- craters it to 0.0364. version_C trained write-only attacks on ~60% of
# steps, so this is not a coverage gap.
#
# Take t71 EXACTLY and add the read projections (q/k/v inherit attn.o_proj's tent, gate/up
# inherit mlp.down_proj's). Everything else identical.
#   FULL  arm: isolates the read/write variable alone.
#   plain arm: rank-1 (which fires) uses the plain application, so this separates
#              "read projections matter" from "plain application matters".
#
# If GSM8K craters with +read, MAD is keyed to read-projection ablation and heretic
# structurally cannot trigger it. If it does not crater, the hypothesis is wrong.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/readproj_test.log
echo "=== read-projection test $(date -u) ===" | tee "$LOG"
echo "=== ref: t71 write-only 0.3231 harm / GSM8K 0.3859 | rank-1 read+write GSM8K 0.0364 ===" | tee -a "$LOG"

run () {  # $1=tag  $2=application
  local TAG=$1
  local APP=$2
  local D="outputs/${TAG}"
  echo "########## ${TAG} (application=${APP}) ##########" | tee -a "$LOG"
  [ -d "$D" ] || python -u experiments/version_c_replay.py \
      --checkpoint outputs/version_c_qwen_500.pt --trial t71 \
      --params-json results/heretic_vc_trials.json \
      --direction-recipe heretic --application "$APP" --add-read-proj \
      --out "$D" 2>&1 | tail -4 | tee -a "$LOG"
  [ -d "results/${TAG}" ] || python -u experiments/p0_baseline_eval.py \
    --run-id "$TAG" --model-id "$D" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking off >>"$LOG" 2>&1
  pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true; sleep 5
  [ -d "results/${TAG}_judged" ] || python -u experiments/judge_generations.py \
    --generations "results/${TAG}/generations.jsonl" --run-id "${TAG}_judged" \
    --num-workers 32 >>"$LOG" 2>&1
  lm_eval --model vllm \
    --model_args "pretrained=${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45" \
    --tasks gsm8k --num_fewshot 5 --batch_size auto \
    --output_path "results/${TAG}_gsm8k" >>"$LOG" 2>&1
  python - "$TAG" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, glob
t = sys.argv[1]
s = json.load(open("results/%s_judged/summary.json" % t)); u = s["usefulness_counts"]; n = s["n"]
g = "pend"
for f in glob.glob("results/%s_gsm8k/**/results_*.json" % t, recursive=True):
    r = json.load(open(f))["results"]
    if "gsm8k" in r: g = "%.4f" % r["gsm8k"]["exact_match,strict-match"]
print("  -> harm %.4f  gib %.4f  ref %.4f  GSM8K %s   (clean 0.3844)" % (
    u.get("harmful_actionable",0)/n, u.get("gibberish",0)/n, u.get("refused",0)/n, g))
PY
  rm -rf "$D"
}

run rp_t71_full  heretic_full
run rp_t71_plain plain
echo "=== READPROJ DONE $(date -u) ===" | tee -a "$LOG"
