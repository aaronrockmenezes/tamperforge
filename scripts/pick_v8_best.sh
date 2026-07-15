#!/usr/bin/env bash
# Pick the best v8 ckpt from --save-every intermediates (training oscillates through the
# clean<->wall Pareto). For each <stem>.s<step>.pt: materialize ATTACKED -> AdvBench 200 gens
# (JUDGE LOCALLY, want low harmAct) + CLEAN -> ifeval probe (want high). Then pick the ckpt
# with lowest attacked-harmAct AND high clean-probe.
#   MID=meta-llama/Llama-3.2-1B-Instruct STEM=outputs/tamper_resistant_llama32_1b_v8.pt DL=13 \
#     CUDA_VISIBLE_DEVICES=0 bash scripts/pick_v8_best.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
MID="${MID:?set MID}"; STEM="${STEM:?set STEM (the --out path)}"; DL="${DL:?set DL}"
QWEN_THINKING="${QWEN_THINKING:-${TF_QWEN_THINKING:-off}}"
export TF_QWEN_THINKING="$QWEN_THINKING"
export TF_IFEVAL_MAX_NEW="${TF_IFEVAL_MAX_NEW:-48}"
VLLM_TEMPERATURE="${VLLM_TEMPERATURE:-0.0}"
VLLM_TOP_P="${VLLM_TOP_P:-1.0}"
VLLM_TOP_K="${VLLM_TOP_K:--1}"
VLLM_PRESENCE_PENALTY="${VLLM_PRESENCE_PENALTY:-0.0}"
VLLM_MAX_NEW_TOKENS="${VLLM_MAX_NEW_TOKENS:-512}"
VLLM_MAX_LENGTH="${VLLM_MAX_LENGTH:-4096}"
VLLM_BATCH_SIZE="${VLLM_BATCH_SIZE:-64}"
for ck in "${STEM}".s*.pt "${STEM}"; do
  [ -f "$ck" ] || continue
  tag=$(basename "$ck" .pt | sed 's#.*resistant_##')
  echo "### $tag -> $ck ###"
  # attacked: wall check (judge advbench locally)
  $PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$ck" --attack all \
      --direction-layer "$DL" --qwen-thinking "$QWEN_THINKING" --out /workspace/outputs/_pk_att
  $PY experiments/p0_baseline_eval.py --backend vllm --model-id /workspace/outputs/_pk_att \
      --prompt-source advbench --advbench-source walledai --n-prompts 200 --max-new-tokens "$VLLM_MAX_NEW_TOKENS" \
      --max-length "$VLLM_MAX_LENGTH" --vllm-batch-size "$VLLM_BATCH_SIZE" --n-arc 0 --qwen-thinking "$QWEN_THINKING" \
      --vllm-temperature "$VLLM_TEMPERATURE" --vllm-top-p "$VLLM_TOP_P" --vllm-top-k "$VLLM_TOP_K" \
      --vllm-presence-penalty "$VLLM_PRESENCE_PENALTY" --run-id "pk_${tag}_att_adv200"
  # clean: capability probe (printed inline) + clean AdvBench gens (clean-harm = axis #1, judge locally)
  $PY experiments/save_p1b_checkpoint.py --model-id "$MID" --checkpoint "$ck" --attack none \
      --direction-layer "$DL" --qwen-thinking "$QWEN_THINKING" --out /workspace/outputs/_pk_clean
  $PY - <<EOF
import sys, json, pathlib; sys.path.insert(0,"experiments")
from train_tamper_resistant_v8 import _clean_ifeval_probe, _IFEVAL_PROBE
from tamperforge import load_model
m,t,d=load_model("/workspace/outputs/_pk_clean","cuda")
v=_clean_ifeval_probe(m,t,d); N=len(_IFEVAL_PROBE)
print(f"  [CLEAN probe] $tag: {v:.3f} ({N} prompts)")
p=pathlib.Path("results/pk_clean_probes.jsonl")
with open(p,"a") as f: f.write(json.dumps({"tag":"$tag","clean_probe":v,"n":N})+"\n")
EOF
  $PY experiments/p0_baseline_eval.py --backend vllm --model-id /workspace/outputs/_pk_clean \
      --prompt-source advbench --advbench-source walledai --n-prompts 200 --max-new-tokens "$VLLM_MAX_NEW_TOKENS" \
      --max-length "$VLLM_MAX_LENGTH" --vllm-batch-size "$VLLM_BATCH_SIZE" --n-arc 0 --qwen-thinking "$QWEN_THINKING" \
      --vllm-temperature "$VLLM_TEMPERATURE" --vllm-top-p "$VLLM_TOP_P" --vllm-top-k "$VLLM_TOP_K" \
      --vllm-presence-penalty "$VLLM_PRESENCE_PENALTY" --run-id "pk_${tag}_clean_adv200"
  rm -rf /workspace/outputs/_pk_att /workspace/outputs/_pk_clean
done
echo "### done — judge results/pk_*_{att,clean}_adv200 locally. 4-axis pick: snapshot with"
echo "    LOW clean-harm (#1) + high CLEAN probe (#2) + LOW att-harm (#3, wall) + low att-probe (#4). ###"
