#!/usr/bin/env bash
# Judge the heretic-vs-base-Llama winners. ceiling_llama.sh ran the 200-trial study but never
# replayed its winners, so the Llama ceiling is currently defined by surgical (0.6923) rather
# than by the strongest attacker. Gated on reevc so only one vLLM is resident at a time.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/hbase_llama.log
echo "=== waiting for version_C seed 1 to release the GPU ===" | tee "$LOG"
while ! grep -aq "VC SEED1 DONE" logs/training_runs/reeval_vc.log 2>/dev/null; do sleep 60; done
sleep 30
echo "=== heretic-vs-base-Llama winners $(date -u) ===" | tee -a "$LOG"

python - <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
sys.path.insert(0, "experiments")
from version_c_loop import parse_trials, pick_winners
t = parse_trials(open("logs/training_runs/heretic_lbase.log", encoding="utf-8", errors="replace").read())
w = pick_winners(t, k=2, kl_max=0.5)
json.dump({"heretic_trials": {"t%d" % x["trial"]: x for x in w}},
          open("results/heretic_lbase_trials.json", "w"), indent=2)
print("parsed %d trials; winners %s" % (len(t), [(x["trial"], x["refusals"], x["kl"]) for x in w]))
PY

for T in $(python -c "import json;print(' '.join(json.load(open('results/heretic_lbase_trials.json'))['heretic_trials']))"); do
  D="outputs/hlbase_${T}"
  echo "########## base Llama + heretic ${T} ##########" | tee -a "$LOG"
  # NOTE: --model-id, no --checkpoint => attacks the BASE model
  [ -f "$D/model.safetensors" ] || python -u experiments/version_c_replay.py \
    --model-id meta-llama/Llama-3.2-1B-Instruct --trial "$T" \
    --params-json results/heretic_lbase_trials.json \
    --direction-recipe heretic --application heretic_full --out "$D" 2>&1 | tail -3 | tee -a "$LOG"
  [ -f "results/hlbase_${T}/generations.jsonl" ] || python -u experiments/p0_baseline_eval.py \
    --run-id "hlbase_${T}" --model-id "$D" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 >>"$LOG" 2>&1
  [ -f "results/hlbase_${T}_judged/summary.json" ] || python -u experiments/judge_generations.py \
    --generations "results/hlbase_${T}/generations.jsonl" --run-id "hlbase_${T}_judged" \
    --num-workers 32 >>"$LOG" 2>&1
  find "results/hlbasecap_${T}_gsm8k" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . || \
    lm_eval --model vllm --model_args "pretrained=${D},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45" \
      --tasks gsm8k --num_fewshot 5 --batch_size auto --output_path "results/hlbasecap_${T}_gsm8k" >>"$LOG" 2>&1
  python - "hlbase_${T}" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, glob, os
t = sys.argv[1]
p = "results/%s_judged/summary.json" % t
if not os.path.exists(p): print("  FAILED: no summary"); raise SystemExit(1)
s = json.load(open(p)); u, n = s["usefulness_counts"], s["n"]
g = "-"
for f in glob.glob("results/%s_gsm8k/**/results_*.json" % t.replace("hlbase_","hlbasecap_"), recursive=True):
    r = json.load(open(f))["results"]
    if "gsm8k" in r: g = "%.4f" % r["gsm8k"]["exact_match,strict-match"]
print("  n=%d harm %.4f gib %.4f ref %.4f GSM8K %s" % (
    n, u.get("harmful_actionable",0)/n, u.get("gibberish",0)/n, u.get("refused",0)/n, g))
PY
  rm -rf "$D"
done
echo "=== LLAMA HERETIC CEILING DONE $(date -u) ===" | tee -a "$LOG"
