#!/usr/bin/env bash
# Runs after the heretic-vs-base study: (1) judge base's heretic winners to get the
# CEILING, then (2) the version_C direction-layer sweep at heretic's operating point.
# Both need the GPU, so this waits rather than contending with the study.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/chain_ceiling.log
echo "=== waiting for heretic-vs-base study ===" | tee "$LOG"
while ! grep -aq "HERETIC-BASE STUDY DONE" logs/training_runs/heretic_base_eval.log 2>/dev/null; do sleep 30; done
sleep 30
echo "=== study done, extracting base winners $(date -u) ===" | tee -a "$LOG"

python - <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
sys.path.insert(0, "experiments")
from version_c_loop import parse_trials, pick_winners
t = parse_trials(open("logs/training_runs/heretic_base_trials.log",
                      encoding="utf-8", errors="replace").read())
w = pick_winners(t, k=2, kl_max=0.5)
json.dump({"heretic_trials": {"t%d" % x["trial"]: x for x in w}},
          open("results/heretic_base_trials.json", "w"), indent=2)
print("parsed %d trials; winners: %s" % (len(t), [(x["trial"], x["refusals"], x["kl"]) for x in w]))
PY

for T in $(python -c "import json;print(' '.join(json.load(open('results/heretic_base_trials.json'))['heretic_trials']))"); do
  D="outputs/hbase_${T}"
  echo "########## base + heretic ${T} ##########" | tee -a "$LOG"
  python -u experiments/version_c_replay.py --trial "$T" \
    --params-json results/heretic_base_trials.json \
    --direction-recipe heretic --application heretic_full --out "$D" 2>&1 | tee -a "$LOG"
  python -u experiments/p0_baseline_eval.py --run-id "hbase_${T}" --model-id "$D" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking off 2>&1 | tee -a "$LOG"
  python -u experiments/judge_generations.py --generations "results/hbase_${T}/generations.jsonl" \
    --run-id "hbase_${T}_judged" --num-workers 32 2>&1 | tee -a "$LOG"
  lm_eval --model vllm \
    --model_args "pretrained=${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45" \
    --tasks gsm8k --num_fewshot 5 --batch_size auto \
    --output_path "results/hbasecap_${T}_gsm8k" 2>&1 | tee -a "$LOG" | grep -aE "\|gsm8k"
  rm -rf "$D"
done
echo "=== CEILING DONE $(date -u) ===" | tee -a "$LOG"

echo "=== starting version_C DL sweep ===" | tee -a "$LOG"
bash dl_sweep_vc.sh 2>&1 | tail -5 | tee -a "$LOG"
echo "=== ALL DONE $(date -u) ===" | tee -a "$LOG"
