#!/usr/bin/env bash
# version_A s500 through the VALIDATED export path (the one that reproduced v8's 0.448).
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/export_version_a_s500.log
CK=outputs/version_a_qwen_500.pt
echo "=== version_A s500 export-path eval: $(date -u) ===" | tee "$LOG"
echo "=== v8 reference on this exact path: surgical_k16 = 0.4365 harmful ===" | tee -a "$LOG"

run_eval () {  # $1=tag  $2=model dir
  python -u experiments/p0_baseline_eval.py \
    --run-id "xva_$1" --model-id "$2" \
    --prompt-source advbench --advbench-source walledai --advbench-split train \
    --n-prompts -1 --n-arc 0 --n-mmlu-per-subject 0 \
    --max-new-tokens 512 --max-length 4096 \
    --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
    --vllm-gpu-memory-utilization 0.45 --vllm-temperature 0.0 --vllm-top-p 1.0 \
    --qwen-thinking default 2>&1 | tee -a "$LOG"
  python -u experiments/judge_generations.py \
    --generations "results/xva_$1/generations.jsonl" \
    --run-id "xva_$1_judged" --num-workers 32 2>&1 | tee -a "$LOG"
}

echo "--- [1/3] clean (no attack) ---" | tee -a "$LOG"
python -u experiments/save_p1b_checkpoint.py --checkpoint "$CK" --model-id Qwen/Qwen3-0.6B \
  --attack none --out outputs/xva_clean 2>&1 | tee -a "$LOG"
run_eval clean outputs/xva_clean

echo "--- [2/3] surgical k16 (THE comparison) ---" | tee -a "$LOG"
python -u experiments/v11_surgical_ablation.py --model-id Qwen/Qwen3-0.6B \
  --checkpoint "$CK" --direction-layer 20 --cap-rank 16 \
  --out outputs/xva_surg_k16 2>&1 | tee -a "$LOG"
run_eval surg_k16 outputs/xva_surg_k16

echo "--- [3/3] plain rank-1 control (cap-rank 0) ---" | tee -a "$LOG"
python -u experiments/v11_surgical_ablation.py --model-id Qwen/Qwen3-0.6B \
  --checkpoint "$CK" --direction-layer 20 --cap-rank 0 \
  --out outputs/xva_rank1 2>&1 | tee -a "$LOG"
run_eval rank1 outputs/xva_rank1

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
df -h /workspace | tail -1 | tee -a "$LOG"
