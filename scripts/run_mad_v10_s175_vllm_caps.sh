#!/usr/bin/env bash
# Capability eval for MAD-v10 S175 exported vLLM variants.
# Runs clean trained, attacked trained, and attacked base through the same suites.
set -uo pipefail
cd "$(dirname "$0")/.."

source /venv/main/bin/activate 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXPORT_ROOT="${EXPORT_ROOT:-outputs/mad_v10_s175_rank1_vllm_20260726_132333}"
RUN_ID="${RUN_ID:-mad_v10_s175_vllm_caps_$(date +%Y%m%d_%H%M%S)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM="${GPU_MEM:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
EVAL_LIMIT="${EVAL_LIMIT:-200}"
MMLU_LIMIT_PER_TASK="${MMLU_LIMIT_PER_TASK:-40}"
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

result_exists() {
  find "$1" -type f -name 'results_*.json' -print -quit 2>/dev/null | grep -q .
}

run_suite() {
  local variant="$1"
  local model_path="$2"
  local suite="$3"
  local tasks="$4"
  local limit="$5"
  shift 5

  local out="results/${RUN_ID}/${variant}/${suite}"
  local args="pretrained=${model_path},dtype=${DTYPE},trust_remote_code=True,max_model_len=${MAX_MODEL_LEN},gpu_memory_utilization=${GPU_MEM},max_num_seqs=${MAX_NUM_SEQS},enable_thinking=false"

  if result_exists "${out}"; then
    echo "[skip] ${variant}/${suite}: result exists"
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
    --limit "${limit}" \
    --output_path "${out}" \
    "$@" &
  local eval_pid=$!

  # Some vLLM versions hang after lm-eval has already written the final result.
  # Once that file is safely present, stop only this evaluator's EngineCore so
  # lm-eval exits and the next suite can start.
  while kill -0 "${eval_pid}" 2>/dev/null; do
    if result_exists "${out}"; then
      sleep 5
      local engine_pid
      engine_pid="$(pgrep -P "${eval_pid}" -f "VLLM::EngineCore" | head -1 || true)"
      if [ -n "${engine_pid}" ]; then
        echo "[cleanup] ${variant}/${suite}: stopping finished EngineCore ${engine_pid}"
        kill -TERM "${engine_pid}" 2>/dev/null || true
      fi
      break
    fi
    sleep 2
  done

  wait "${eval_pid}"
  local status=$?
  if result_exists "${out}"; then
    status=0
  fi
  cleanup_vllm_leftovers
  return "${status}"
}

for variant in trained_clean trained_attacked base_attacked; do
  model_path="${EXPORT_ROOT}/${variant}"
  if [ ! -d "${model_path}" ]; then
    echo "!! missing ${model_path}; skipping ${variant}"
    continue
  fi

  run_suite "${variant}" "${model_path}" ifeval ifeval "${EVAL_LIMIT}" \
    --num_fewshot 0 \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" "temperature=0"

  run_suite "${variant}" "${model_path}" gsm8k gsm8k "${EVAL_LIMIT}" \
    --num_fewshot 5 \
    --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" "temperature=0"

  run_suite "${variant}" "${model_path}" arc_challenge arc_challenge "${EVAL_LIMIT}" \
    --num_fewshot 0

  run_suite "${variant}" "${model_path}" mmlu_5x40 "${MMLU_TASKS}" "${MMLU_LIMIT_PER_TASK}" \
    --num_fewshot 0

  if task_exists "${AGI_TASK}"; then
    run_suite "${variant}" "${model_path}" agieval_200 "${AGI_TASK}" "${EVAL_LIMIT}" \
      --num_fewshot 0 \
      --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" "temperature=0"
  else
    echo "[skip] ${variant}/agieval_200: task '${AGI_TASK}' not installed"
  fi

  if task_exists reclor; then
    run_suite "${variant}" "${model_path}" reclor reclor "${EVAL_LIMIT}" \
      --num_fewshot 0
  else
    echo "[skip] ${variant}/reclor: task 'reclor' not installed"
  fi
done

echo "### done: results/${RUN_ID}"
