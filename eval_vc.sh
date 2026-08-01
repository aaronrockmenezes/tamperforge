#!/usr/bin/env bash
# version_B s500 through the validated export path -- head-to-head with version_A.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/eval_version_c.log
CK=outputs/version_c_qwen_500.pt
MMLU="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_machine_learning,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics,mmlu_computer_security"
echo "=== version_C s500 eval: $(date -u) ===" | tee "$LOG"
echo "=== version_B reference: clean .0019/.9962ref, rank1 .0000/GSM.0091, surg16 .0000/GSM.1054/gib.9808 ===" | tee -a "$LOG"

run () {  # $1=tag $2=dir
  python -u experiments/p0_baseline_eval.py --run-id "xvc_$1" --model-id "$2" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
    --max-new-tokens 512 --max-length 4096 --backend vllm --vllm-batch-size 64 \
    --vllm-dtype bfloat16 --vllm-gpu-memory-utilization 0.45 \
    --vllm-temperature 0.0 --vllm-top-p 1.0 --qwen-thinking off 2>&1 | tee -a "$LOG"
  python -u experiments/judge_generations.py --generations "results/xvc_$1/generations.jsonl" \
    --run-id "xvc_$1_judged" --num-workers 32 2>&1 | tee -a "$LOG"
  a="pretrained=${2},dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.45"
  for spec in "arc:arc_challenge:0" "mmlu:${MMLU}:0" "gsm8k:gsm8k:5"; do
    nm="${spec%%:*}"; rest="${spec#*:}"; tk="${rest%:*}"; sh="${rest##*:}"
    o="results/xvccap_$1_${nm}"; [ -d "$o" ] && continue
    lm_eval --model vllm --model_args "$a" --tasks "$tk" --num_fewshot "$sh" \
      --batch_size auto --output_path "$o" 2>&1 | tee -a "$LOG" | grep -aE "\|acc|\|exact_match"
  done
}

echo "--- [1/3] clean ---" | tee -a "$LOG"
python -u experiments/save_p1b_checkpoint.py --checkpoint "$CK" --model-id Qwen/Qwen3-0.6B \
  --attack none --out outputs/xvc_clean 2>&1 | tee -a "$LOG"
run clean outputs/xvc_clean

echo "--- [2/3] rank-1 ---" | tee -a "$LOG"
python -u experiments/v11_surgical_ablation.py --model-id Qwen/Qwen3-0.6B --checkpoint "$CK" \
  --direction-layer 20 --cap-rank 0 --out outputs/xvc_rank1 2>&1 | tee -a "$LOG"
run rank1 outputs/xvc_rank1

echo "--- [3/3] surgical k16 ---" | tee -a "$LOG"
python -u experiments/v11_surgical_ablation.py --model-id Qwen/Qwen3-0.6B --checkpoint "$CK" \
  --direction-layer 20 --cap-rank 16 --out outputs/xvc_surg_k16 2>&1 | tee -a "$LOG"
run surg_k16 outputs/xvc_surg_k16

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"

# --- stage 2: fresh 200-trial Heretic against version_C clean -----------------
# 200 trials, NOT the 24 used in-loop: version_B faced a 200-trial attacker, so anything
# shorter compares version_C against a weaker adversary and the head-to-head is meaningless.
# Non-interactive (stdin closed, fresh checkpoint dir); parameters are parsed from the log
# afterwards, heretic's own save path is unused.
echo "--- [4/4] fresh heretic 200 trials vs version_C clean ---" | tee -a "$LOG"
rm -rf /tmp/hcp_vc
heretic --model outputs/xvc_clean --n-trials 200 --seed 0 \
  --study-checkpoint-dir /tmp/hcp_vc < /dev/null \
  > logs/training_runs/heretic_version_c.log 2>&1
echo "heretic rc=$? (rc=1 expected: EOF at its menu; trials already logged)" | tee -a "$LOG"
echo "trials logged: $(grep -ac 'Running trial' logs/training_runs/heretic_version_c.log)" | tee -a "$LOG"
echo "=== EVAL DONE $(date -u) ===" | tee -a "$LOG"
