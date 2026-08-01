#!/usr/bin/env bash
# Repair the two base-Llama arms lost to the pkill incident.
#
# Root cause of BOTH: the resume guard tested `[ -d results/<tag> ]`, i.e. the DIRECTORY,
# not whether the run actually produced anything. A partial dir left by the killed job made
# the rerun skip generation. Guards must test for the artifact (generations.jsonl /
# summary.json), never the directory.
#   clean : dir present, 0 generations  -> needs full regeneration (GPU, waits for heretic)
#   rank1 : 520 generations, never judged -> judge only (API, safe to run now)
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/fix_lbase.log
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
echo "=== repair base-Llama arms $(date -u) ===" | tee "$LOG"

# 1. rank1: judge only -- no GPU, run immediately
if [ ! -f results/lbase_rank1_judged/summary.json ]; then
  echo "--- judging lbase_rank1 (generations already present) ---" | tee -a "$LOG"
  rm -rf results/lbase_rank1_judged
  python -u experiments/judge_generations.py --generations results/lbase_rank1/generations.jsonl \
    --run-id lbase_rank1_judged --num-workers 32 >>"$LOG" 2>&1
  python - lbase_rank1 <<'PY' 2>&1 | tee -a "$LOG"
import json,sys
s=json.load(open("results/%s_judged/summary.json"%sys.argv[1])); u=s["usefulness_counts"]; n=s["n"]
print("  rank1 -> n=%d harm %.4f gib %.4f ref %.4f"%(n,u.get("harmful_actionable",0)/n,
      u.get("gibberish",0)/n,u.get("refused",0)/n))
PY
fi

# 2. clean: needs the GPU -- wait for the heretic study to finish
echo "--- waiting for heretic-vs-base-Llama before regenerating clean ---" | tee -a "$LOG"
while ! grep -aq "LLAMA CEILING DONE" logs/training_runs/ceiling_llama.log 2>/dev/null; do sleep 60; done
sleep 30

if [ ! -f results/lbase_clean/generations.jsonl ]; then
  echo "--- regenerating lbase_clean ---" | tee -a "$LOG"
  rm -rf results/lbase_clean results/lbase_clean_judged
  python -u experiments/p0_baseline_eval.py --run-id lbase_clean --model-id outputs/lbase_clean_hf \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 >>"$LOG" 2>&1
  python -u experiments/judge_generations.py --generations results/lbase_clean/generations.jsonl \
    --run-id lbase_clean_judged --num-workers 32 >>"$LOG" 2>&1
fi
for spec in "arc:arc_challenge:0" "mmlu:${MMLU}:0" "gsm8k:gsm8k:5"; do
  nm="${spec%%:*}"; rest="${spec#*:}"; tk="${rest%:*}"; sh="${rest##*:}"
  o="results/lbasecap_clean_${nm}"
  find "$o" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && continue
  rm -rf "$o"
  lm_eval --model vllm --model_args "pretrained=outputs/lbase_clean_hf,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45" \
    --tasks "$tk" --num_fewshot "$sh" --batch_size auto --output_path "$o" >>"$LOG" 2>&1
done
python - lbase_clean <<'PY' 2>&1 | tee -a "$LOG"
import json,sys,glob
s=json.load(open("results/%s_judged/summary.json"%sys.argv[1])); u=s["usefulness_counts"]; n=s["n"]
g="-"
for f in glob.glob("results/lbasecap_clean_gsm8k/**/results_*.json",recursive=True):
    r=json.load(open(f))["results"]
    if "gsm8k" in r: g="%.4f"%r["gsm8k"]["exact_match,strict-match"]
print("  clean -> n=%d harm %.4f gib %.4f ref %.4f GSM8K %s"%(n,u.get("harmful_actionable",0)/n,
      u.get("gibberish",0)/n,u.get("refused",0)/n,g))
PY
echo "=== FIX DONE $(date -u) ===" | tee -a "$LOG"
