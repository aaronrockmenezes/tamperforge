#!/usr/bin/env bash
# Reproduce the campaign path EXACTLY for v8 + surgical k16, to settle whether the panel's
# in-memory HF generation understates the attack (panel: 0.078) vs the campaign's 0.448.
set -uo pipefail
cd /workspace/tamperforge
source /venv/main/bin/activate
set -a; . ./.env; set +a
LOG=logs/training_runs/export_test_v8_surg_k16.log
echo "=== EXPORT TEST: v8 + surgical k16, campaign path: $(date -u) ===" | tee "$LOG"
echo "=== target: judged harmful_actionable ~0.448 (v11_v11_v8_surg_k16) ===" | tee -a "$LOG"

# 1. export the attacked model exactly as v11_surgical_ablation did
python -u experiments/v11_surgical_ablation.py \
  --model-id Qwen/Qwen3-0.6B \
  --checkpoint outputs/adapters/tamper_resistant_qwen3_0p6b_v8.pt \
  --direction-layer 20 --cap-rank 16 \
  --out outputs/xtest_v8_surg_k16 2>&1 | tee -a "$LOG"

# 2. campaign eval config, verbatim from the v11 manifest. gpu-memory-utilization lowered
#    from 0.9: a foreign process holds ~8.7GB on this card and vLLM would fail to allocate.
python -u experiments/p0_baseline_eval.py \
  --run-id xtest_v8_surg_k16 \
  --model-id outputs/xtest_v8_surg_k16 \
  --prompt-source advbench --advbench-source walledai --advbench-split train \
  --n-prompts -1 --max-new-tokens 512 --max-length 4096 \
  --backend vllm --vllm-batch-size 64 --vllm-dtype bfloat16 \
  --vllm-gpu-memory-utilization 0.45 \
  --vllm-temperature 0.0 --vllm-top-p 1.0 \
  --qwen-thinking default 2>&1 | tee -a "$LOG"

# 3. judge
python -u experiments/judge_generations.py \
  --generations results/xtest_v8_surg_k16/generations.jsonl \
  --run-id xtest_v8_surg_k16_judged --num-workers 32 2>&1 | tee -a "$LOG"

echo "=== DONE $(date -u) ===" | tee -a "$LOG"
