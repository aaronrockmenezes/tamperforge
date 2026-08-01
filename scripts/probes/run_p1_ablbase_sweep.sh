#!/usr/bin/env bash
# P1 ablated-base (product) sweep — THE defense test.
#
# Clean-base sweep already proved the entanglement MECHANISM (ablating adapter
# W_out dirs costs capability that random dirs don't: PPL 9513 vs 24 at k=32).
# But cleanbase safety never dropped because the base kept its native refusal.
#
# This sweep strips native refusal first (--adapter-base native_ablated), so the
# adapter is the SOLE safety. Question: as we ablate k adapter dirs, does safety
# (judge ASR) rise ONLY once capability (PPL) is already destroyed? If so ->
# uncensoring costs capability -> defense holds.
#
# Run on the GPU box from repo root:  bash scripts/run_p1_ablbase_sweep.sh
# Then judge the generations LOCALLY (see TODO.md section 3) — keyword ASR lies
# on broken models; judge is the real metric.

set -euo pipefail

ADAPTER=outputs/safety_adapter_p1_ablbase.pt
COMMON=(
  --adapter "$ADAPTER"
  --adapter-base native_ablated
  --model-id google/gemma-3-1b-it
  --advbench-source walledai
  --adapter-layer 13 --direction-layer 13 --abliterate-layers all
  --n-direction 256 --n-advbench 50 --n-arc 100
  --max-new-tokens 128 --no-judge
)

# Reference conditions once (base / base_ablated / base_adapter on the ablated base).
python experiments/p1_mad_crux.py "${COMMON[@]}" \
  --conditions base,base_ablated,base_adapter \
  --run-id p1_ablbase_reference

# Rank sweep: only the k-dependent conditions.
for k in 1 2 4 8 16 32 64; do
  python experiments/p1_mad_crux.py "${COMMON[@]}" \
    --adapter-attack-rank "$k" \
    --conditions base_adapter_ablated_full,base_ablated_randN \
    --run-id "p1_ablbase_rank$k"
done

echo "[done] ablbase sweep complete. Copy results local + judge (TODO.md 1 & 3)."
