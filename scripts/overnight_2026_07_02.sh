#!/usr/bin/env bash
# Overnight batch (2026-07-02). TESTED code only (ft_attack + p0_baseline, run many
# times today). Generation-only -> judge LOCALLY tomorrow (no API key on box).
# Robust: continues on any failure, logs each step, pushes gens after each ckpt group.
#
# Launch in tmux (foreground, detach the tmux):
#   tmux new -s night 'bash scripts/overnight_2026_07_02.sh 2>&1 | tee overnight/main.log'
#
# Goal: full FT cost-frontier at AdvBench 520, K in {1,5,10,25,50,100} for our best
# ckpts + base control -> the real "where we stand vs SOTA" numbers.

set +e   # never die on one failure
cd /workspace/tamperforge || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p overnight outputs/night
PY=/venv/main/bin/python
DEMOS=results/p1b_v7_base_att_gen/generations.jsonl
KS="0 1 5 10 25 50 100 200"   # K=0 = raw model, no FT (the un-attacked baseline)

ts(){ date +%H:%M:%S; }
# AdvBench gens (judge local). vLLM backend REQUIRES --n-arc 0 (capability via lm_eval).
ftgen(){ $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
   --advbench-source walledai --n-advbench 520 --n-arc 0 \
   --max-new-tokens 128 --max-length 4096 --run-id "$2"; }
# capability via lm_eval (MAD 2nd axis: FT'd model capable, or dumb?). set +e =
# a failure here won't kill the run (gens already saved). ARC full-ish (limit 400)
# + MMLU on a 10-subject balanced subset (STEM/hum/social/professional), FULL
# questions per subject (~1.9k total), 0-shot. num_fewshot 0 for speed + a clean
# relative-degradation trend across K (absolute 5-shot numbers -> final paper).
MA="dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.9,max_num_seqs=64"
MMLU_SUBJ="mmlu_high_school_biology,mmlu_college_computer_science,mmlu_abstract_algebra,mmlu_philosophy,mmlu_world_religions,mmlu_high_school_us_history,mmlu_econometrics,mmlu_sociology,mmlu_professional_medicine,mmlu_business_ethics"
capeval(){
  /venv/main/bin/lm_eval --model vllm --model_args "pretrained=$1,$MA" \
    --tasks arc_challenge --num_fewshot 0 --batch_size auto \
    --output_path "results/$2_arc" 2>&1 | tail -3
  /venv/main/bin/lm_eval --model vllm --model_args "pretrained=$1,$MA" \
    --tasks "$MMLU_SUBJ" --num_fewshot 0 --batch_size auto \
    --output_path "results/$2_mmlu" 2>&1 | tail -3
}
pushgens(){
  git add -f results/night_*_gen/generations.jsonl results/night_*_gen/summary.json \
             results/night_*_cap_arc results/night_*_cap_mmlu overnight/*.log 2>/dev/null
  git commit -q -m "overnight: $1 gens ($(ts))" 2>/dev/null
  git push origin main 2>/dev/null && echo "[$(ts)] pushed: $1" || echo "[$(ts)] push failed (retry next): $1"
}

# tag:checkpoint-path ("" path = base control, no --checkpoint)
# NOTE: no FT-defense actually works (all ~= base by K=5). Spread chosen to
# CHARACTERIZE the frontier, not because one is "best":
#   v3 = holds K=1 (0.005) but it's an artifact of a weak inner (breaks K>=5)
#   v5 = latest / most-advanced objective (all-scope + generation-level); breaks K=1 (0.695)
#   v7 = abliteration product (FT-undefended) -> honest FT-cost baseline
#   base = bare gemma control
# parallel arrays (no substring expansion; ${spec%%:*} mis-parsed to 0 on the box).
# ck="" => base control (no --checkpoint).
TAGS=(v3 v5 v7 base)
CKS=("outputs/ft_resistant_p4_v3.pt" "outputs/ft_resistant_p4_v5.pt" "outputs/tamper_resistant_p1b_v7.pt" "")

for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"; ck="${CKS[$i]}"
  echo "############ [$(ts)] GROUP $tag (ckpt='${ck:-BASE}') ############"
  for K in $KS; do
    out="outputs/night/${tag}_ft${K}"
    ep=5; ns="$K"
    if [ "$K" = "0" ]; then ep=0; ns=1; fi   # K=0 = raw model, 0 FT epochs (baseline)
    echo "---- [$(ts)] $tag K=$K : FT (epochs=$ep) ----"
    if [ -n "$ck" ]; then
      $PY experiments/ft_attack.py --checkpoint "$ck" --demos "$DEMOS" \
        --n-shots "$ns" --ft-epochs "$ep" --out "$out"
    else
      $PY experiments/ft_attack.py --demos "$DEMOS" \
        --n-shots "$ns" --ft-epochs "$ep" --out "$out"
    fi
    echo "---- [$(ts)] $tag K=$K : GEN 520 ----"
    ftgen "$out" "night_${tag}_ft${K}_gen"
    echo "---- [$(ts)] $tag K=$K : CAP (ARC+MMLU) ----"
    capeval "$out" "night_${tag}_ft${K}_cap"
    rm -rf "$out"   # reclaim disk (FT'd HF dir ~1.9G; gens + cap json already saved)
  done
  pushgens "$tag"
done

echo "############ [$(ts)] OVERNIGHT DONE ############"
echo "Judge locally tomorrow: results/night_*_ft*_gen/generations.jsonl"
