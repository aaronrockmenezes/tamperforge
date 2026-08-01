#!/usr/bin/env bash
# BASE Llama-3.2-1B-Instruct ceiling control -- the denominator for any Llama defense number.
#
# Without this, "heretic gets X% of headroom" cannot be computed for this architecture. Qwen's
# ceiling was 0.6596 harmful with capability intact; Llama's is expected LOWER because its
# safety is more diffuse (devlog_2026_07_02: base uncensors to only 0.60-0.66 even at its best
# layer vs gemma's 0.82).
#
# --direction-layer 13, NOT a scaled-from-Qwen guess. Measured: the base ablation sweep across
# all 16 layers peaks at L13 (0.5650 judged harmful); L11 is a local MINIMUM at 0.1950, and
# L15 is dead at 0.0700. devlog_2026_07_02 records that a mid-depth guess (L8) cost 0.654 ->
# 0.133 attacked harm until it was corrected to L13.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/ceiling_llama.log
MID="meta-llama/Llama-3.2-1B-Instruct"
DL=13
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
echo "=== base Llama-3.2-1B ceiling control (DL=${DL}) $(date -u) ===" | tee "$LOG"
echo "=== Qwen reference: clean 0.2577 | rank-1 0.6385 | surgical 0.6962 | heretic 0.6596 ===" | tee -a "$LOG"

run () {  # $1=tag $2=model
  local TAG=$1
  local MD=$2
  echo "########## ${TAG} ##########" | tee -a "$LOG"
  [ -d "results/lbase_${TAG}" ] || python -u experiments/p0_baseline_eval.py \
    --run-id "lbase_${TAG}" --model-id "$MD" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.25 --vllm-temperature 0.0 --vllm-top-p 1.0 >>"$LOG" 2>&1
  # (global pkill removed: it killed concurrent lanes 2026-08-01)
  [ -d "results/lbase_${TAG}_judged" ] || python -u experiments/judge_generations.py \
    --generations "results/lbase_${TAG}/generations.jsonl" --run-id "lbase_${TAG}_judged" \
    --num-workers 32 >>"$LOG" 2>&1
  local a="pretrained=${MD},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.25"
  for spec in "arc:arc_challenge:0" "mmlu:${MMLU}:0" "gsm8k:gsm8k:5"; do
    local nm="${spec%%:*}"; local rest="${spec#*:}"; local tk="${rest%:*}"; local sh="${rest##*:}"
    local o="results/lbasecap_${TAG}_${nm}"; [ -d "$o" ] && continue
    lm_eval --model vllm --model_args "$a" --tasks "$tk" --num_fewshot "$sh" \
      --batch_size auto --output_path "$o" >>"$LOG" 2>&1 &
    local pid=$!; local waited=0
    while kill -0 "$pid" 2>/dev/null; do
      if find "$o" -name "results_*.json" -print -quit 2>/dev/null | grep -q .; then
        sleep 5
        local ep; ep="$(pgrep -P "$pid" -f "VLLM::EngineCore" | head -1 || true)"
        [ -n "$ep" ] && kill -TERM "$ep" 2>/dev/null || true
        sleep 8; kill -TERM "$pid" 2>/dev/null || true; break
      fi
      sleep 10; waited=$((waited+10))
      [ "$waited" -gt 1800 ] && { kill -TERM "$pid" 2>/dev/null; break; }
    done
    wait "$pid" 2>/dev/null || true
  # (global pkill removed: it killed concurrent lanes 2026-08-01)
  done
  python - "lbase_${TAG}" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys, glob
t = sys.argv[1]
s = json.load(open("results/%s_judged/summary.json" % t)); u = s["usefulness_counts"]; n = s["n"]
g = "pend"
for f in glob.glob("results/%s_gsm8k/**/results_*.json" % t.replace("lbase_", "lbasecap_"), recursive=True):
    r = json.load(open(f))["results"]
    if "gsm8k" in r: g = "%.4f" % r["gsm8k"]["exact_match,strict-match"]
print("  -> harm %.4f  gib %.4f  ref %.4f  GSM8K %s" % (
    u.get("harmful_actionable",0)/n, u.get("gibberish",0)/n, u.get("refused",0)/n, g))
PY
}

# materialise base once (heretic needs a local dir)
[ -d outputs/lbase_clean_hf ] || python - <<'PY' 2>&1 | tee -a "$LOG"
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", dtype="bfloat16")
t = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
m.save_pretrained("outputs/lbase_clean_hf", safe_serialization=True)
t.save_pretrained("outputs/lbase_clean_hf")
print("[base] materialised")
PY

run clean outputs/lbase_clean_hf

echo "########## rank-1 (DL=${DL}) ##########" | tee -a "$LOG"
[ -d outputs/lbase_rank1 ] || python -u experiments/v11_surgical_ablation.py --model-id "$MID" \
  --direction-layer $DL --cap-rank 0 --out outputs/lbase_rank1 2>&1 | tail -3 | tee -a "$LOG"
run rank1 outputs/lbase_rank1

echo "########## surgical k16 (DL=${DL}) ##########" | tee -a "$LOG"
[ -d outputs/lbase_surg_k16 ] || python -u experiments/v11_surgical_ablation.py --model-id "$MID" \
  --direction-layer $DL --cap-rank 16 --out outputs/lbase_surg_k16 2>&1 | tail -3 | tee -a "$LOG"
run surg_k16 outputs/lbase_surg_k16

echo "########## heretic 200 trials vs base Llama ##########" | tee -a "$LOG"
rm -rf /tmp/hcp_lbase
heretic --model outputs/lbase_clean_hf --n-trials 200 --seed 0 \
  --study-checkpoint-dir /tmp/hcp_lbase < /dev/null \
  > logs/training_runs/heretic_lbase.log 2>&1
echo "  study rc=$? trials=$(grep -ac 'Running trial' logs/training_runs/heretic_lbase.log)" | tee -a "$LOG"
echo "=== LLAMA CEILING DONE $(date -u) ===" | tee -a "$LOG"
