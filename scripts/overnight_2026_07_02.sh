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
KS="1 5 10 25 50 100 200"

ts(){ date +%H:%M:%S; }
ftgen(){ $PY experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
   --advbench-source walledai --n-advbench 520 --n-arc 0 --max-new-tokens 128 \
   --max-length 4096 --run-id "$2"; }
pushgens(){
  git add -f results/night_*_gen/generations.jsonl overnight/*.log 2>/dev/null
  git commit -q -m "overnight: $1 gens ($(ts))" 2>/dev/null
  git push origin main 2>/dev/null && echo "[$(ts)] pushed: $1" || echo "[$(ts)] push failed (retry next): $1"
}

# tag:checkpoint-path ("" path = base control, no --checkpoint)
GROUPS=(
  "v5:outputs/ft_resistant_p4_v5.pt"        # best FT-defense (all-scope, gen-obj)
  "v7:outputs/tamper_resistant_p1b_v7.pt"   # abliteration PRODUCT -> honest FT-cost baseline
  "base:"                                    # control (bare gemma)
  "v3:outputs/ft_resistant_p4_v3.pt"        # tail: MLP-scope FT-defense, completeness
)

for spec in "${GROUPS[@]}"; do
  tag="${spec%%:*}"; ck="${spec#*:}"
  echo "############ [$(ts)] GROUP $tag (ckpt='${ck:-BASE}') ############"
  for K in $KS; do
    out="outputs/night/${tag}_ft${K}"
    echo "---- [$(ts)] $tag K=$K : FT ----"
    if [ -n "$ck" ]; then
      $PY experiments/ft_attack.py --checkpoint "$ck" --demos "$DEMOS" \
        --n-shots "$K" --ft-epochs 5 --out "$out"
    else
      $PY experiments/ft_attack.py --demos "$DEMOS" \
        --n-shots "$K" --ft-epochs 5 --out "$out"
    fi
    echo "---- [$(ts)] $tag K=$K : GEN 520 ----"
    ftgen "$out" "night_${tag}_ft${K}_gen"
    rm -rf "$out"   # reclaim disk (FT'd HF dir ~1.9G; gens.jsonl already saved)
  done
  pushgens "$tag"
done

echo "############ [$(ts)] OVERNIGHT DONE ############"
echo "Judge locally tomorrow: results/night_*_ft*_gen/generations.jsonl"
