#!/usr/bin/env bash
# version_B s500 through the validated export path -- head-to-head with version_A.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/eval_version_b.log
CK=outputs/version_b_qwen_500.pt
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
echo "=== version_B s500 eval: $(date -u) ===" | tee "$LOG"
echo "=== version_A reference: clean .0058/.9865ref | rank1 .0000/GSM.039 | surg16 .0019 ===" | tee -a "$LOG"

run () {  # $1=tag $2=dir
  python -u experiments/p0_baseline_eval.py --run-id "xvb_$1" --model-id "$2" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
    --max-new-tokens 512 --max-length 4096 --backend vllm --vllm-batch-size 64 \
    --vllm-dtype bfloat16 --vllm-gpu-memory-utilization 0.45 \
    --vllm-temperature 0.0 --vllm-top-p 1.0 --qwen-thinking off 2>&1 | tee -a "$LOG"
  python -u experiments/judge_generations.py --generations "results/xvb_$1/generations.jsonl" \
    --run-id "xvb_$1_judged" --num-workers 32 2>&1 | tee -a "$LOG"
  a="pretrained=${2},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45"
  for spec in "arc:arc_challenge:0" "mmlu:${MMLU}:0" "gsm8k:gsm8k:5"; do
    nm="${spec%%:*}"; rest="${spec#*:}"; tk="${rest%:*}"; sh="${rest##*:}"
    o="results/xvbcap_$1_${nm}"; find "$o" -name 'results_*.json' -print -quit 2>/dev/null | grep -q . && continue
    lm_eval --model vllm --model_args "$a" --tasks "$tk" --num_fewshot "$sh" \
      --batch_size auto --output_path "$o" 2>&1 | tee -a "$LOG" | grep -aE "\|acc|\|exact_match"
  done
}

echo "--- [1/3] clean ---" | tee -a "$LOG"
python -u experiments/save_p1b_checkpoint.py --checkpoint "$CK" --model-id Qwen/Qwen3-0.6B \
  --attack none --out outputs/xvb_clean 2>&1 | tee -a "$LOG"
run clean outputs/xvb_clean

echo "--- [2/3] rank-1 ---" | tee -a "$LOG"
python -u experiments/v11_surgical_ablation.py --model-id Qwen/Qwen3-0.6B --checkpoint "$CK" \
  --direction-layer 20 --cap-rank 0 --out outputs/xvb_rank1 2>&1 | tee -a "$LOG"
run rank1 outputs/xvb_rank1

echo "--- [3/3] surgical k16 ---" | tee -a "$LOG"
python -u experiments/v11_surgical_ablation.py --model-id Qwen/Qwen3-0.6B --checkpoint "$CK" \
  --direction-layer 20 --cap-rank 16 --out outputs/xvb_surg_k16 2>&1 | tee -a "$LOG"
run surg_k16 outputs/xvb_surg_k16

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"
