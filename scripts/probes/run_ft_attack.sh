#!/usr/bin/env bash
# Fine-tuning attack sweep (P4): does FT restore harmful capability?
# For K in {1,5,10,25} harmful demonstrations, FT the tamper-resistant model AND
# the base control, then safety-gen (vLLM). Judge locally -> ASR-vs-K frontier.
# Win for the defense: v7 needs FAR more shots (or never) to reach base's ASR.
#
#   bash scripts/run_ft_attack.sh
set -euo pipefail
cd /workspace/tamperforge
git fetch origin -q && git checkout origin/main -- experiments src scripts docs

CK=outputs/tamper_resistant_p1b_v7.pt
DEMOS=results/p1b_v7_base_att_gen/generations.jsonl   # compliant harmful responses (attacker demos)

ftgen() { python experiments/p0_baseline_eval.py --backend vllm --model-id "$1" \
    --advbench-source walledai --n-advbench 200 --n-arc 0 \
    --max-new-tokens 128 --max-length 4096 --run-id "$2"; }

for K in 1 5 10 25; do
  python experiments/ft_attack.py --checkpoint $CK --demos $DEMOS --n-shots $K --ft-epochs 5 --out outputs/v7_ft$K
  ftgen outputs/v7_ft$K   v7_ft${K}_gen
  python experiments/ft_attack.py               --demos $DEMOS --n-shots $K --ft-epochs 5 --out outputs/base_ft$K
  ftgen outputs/base_ft$K base_ft${K}_gen
done
echo "[ft-attack] done. Judge *_ft*_gen locally; frontier = post-FT ASR vs K, v7 vs base."
