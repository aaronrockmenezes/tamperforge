#!/usr/bin/env bash
# Capability eval for MAD-v10 S175 exported vLLM variants.
# Runs clean trained, attacked trained, and attacked base through the same suites.
set -uo pipefail
cd "$(dirname "$0")/.."

source /venv/main/bin/activate 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXPORT_ROOT="${EXPORT_ROOT:-outputs/mad_v10_s175_rank1_vllm_20260726_132333}"
RUN_ID="${RUN_ID:-mad_v10_s175_vllm_caps_$(date +%Y%m%d_%H%M%S)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM="${GPU_MEM:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
MMLU_TASKS="${MMLU_TASKS:-mmlu_professional_law,mmlu_high_school_biology,mmlu_high_school_us_history,mmlu_high_school_world_history,mmlu_computer_security}"
AGI_TASK="${AGI_TASK:-agieval}"

if command -v lm-eval >/dev/null 2>&1; then
  LM_EVAL=(lm-eval run)
else
  LM_EVAL=(lm_eval run)
fi

mkdir -p "results/${RUN_ID}" logs/evals

task_exists() {
  local task="$1"
  python - "$task" <<'PY'
import sys
from lm_eval.tasks import TaskManager
tm = TaskManager()
needle = sys.argv[1]
names = set(tm.all_tasks)
groups = set(getattr(tm, "_task_index", {}).keys())
sys.exit(0 if needle in names or needle in groups else 1)
PY
}

cleanup_vllm_leftovers() {
  # lm-eval/vLLM can leave an orphaned EngineCore process that keeps almost all
  # VRAM reserved after a suite exits. Kill only that vLLM worker, not Ray/Jupyter.
  pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 5
}

run_suite() {
  local variant="$1"
  local model_path="$2"
  local suite="$3"
  local tasks="$4"
  shift 4

  local out="results/${RUN_ID}/${variant}/${suite}"
  local args="pretrained=${model_path},dtype=${DTYPE},trust_remote_code=True,max_model_len=${MAX_MODEL_LEN},gpu_memory_utilization=${GPU_MEM},max_num_seqs=${MAX_NUM_SEQS}"

  if [ -d "${out}" ]; then
    echo "[skip] ${variant}/${suite}: ${out} exists"
    return 0
  fi

  mkdir -p "$(dirname "${out}")"
  echo "############ ${variant}/${suite} -> ${tasks} ############"
  "${LM_EVAL[@]}" \
    --model vllm \
    --model_args "${args}" \
    --tasks "${tasks}" \
    --batch_size auto \
    --apply_chat_template \
    --output_path "${out}" \
    "$@"
  local status=$?
  cleanup_vllm_leftovers
  return "${status}"
}

for variant in trained_clean trained_attacked base_attacked; do
  model_path="${EXPORT_ROOT}/${variant}"
  if [ ! -d "${model_path}" ]; then
    echo "!! missing ${model_path}; skipping ${variant}"
    continue
  fi

  run_suite "${variant}" "${model_path}" ifeval ifeval \
    --num_fewshot 0 \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" "temperature=0"

  run_suite "${variant}" "${model_path}" gsm8k gsm8k \
    --num_fewshot 5 \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" "temperature=0"

  run_suite "${variant}" "${model_path}" arc_challenge arc_challenge \
    --num_fewshot 0

  run_suite "${variant}" "${model_path}" mmlu_5x200 "${MMLU_TASKS}" \
    --num_fewshot 0 \
    --limit 200

  if task_exists "${AGI_TASK}"; then
    run_suite "${variant}" "${model_path}" agieval_200 "${AGI_TASK}" \
      --num_fewshot 0 \
      --limit 200 \
      --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" "temperature=0"
  else
    echo "[skip] ${variant}/agieval_200: task '${AGI_TASK}' not installed"
  fi

  if task_exists reclor; then
    run_suite "${variant}" "${model_path}" reclor reclor \
      --num_fewshot 0
  else
    echo "[skip] ${variant}/reclor: task 'reclor' not installed"
  fi
done

echo "### done: results/${RUN_ID}"
