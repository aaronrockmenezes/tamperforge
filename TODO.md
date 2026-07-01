# TODO — next session

> Created 2026-07-01 end of session. P0 done + archived. P1 first run done but
> **inconclusive** (direction-count confound). Next = rank sweep to settle it.

## Status recap

- **P0**: complete, judged, committed. base/heretic/extreme baselines in
  `docs/results_2026_07_01.md`.
- **P1 (cleanbase adapter)**: ran, `results/p1_cleanbase_nojudge_v2/`. Confound
  confirmed — ablating 229 *random* dirs destroys the model as badly as ablating
  229 adapter dirs, so the all-229 attack proves nothing. See P1 section in
  `docs/results_2026_07_01.md`.
- **P1 (ablbase adapter)**: run in progress at session end
  (`results/p1_ablbase_nojudge/`), all-rank attack — will also hit the confound;
  needs the same rank sweep.

## 1. Copy remaining results to local (after ablbase proc finishes)

```bash
cd "/Users/aaronrockmenezes/Desktop/Projects/Mech Interp/tamperforge"
for d in p1_ablbase_nojudge; do
  mkdir -p "results/$d"
  scp vast_tamperforge:/workspace/tamperforge/results/$d/generations.jsonl "results/$d/generations.jsonl"
  scp vast_tamperforge:/workspace/tamperforge/results/$d/summary.json "results/$d/summary.json"
done
# after any rank-sweep runs, also:
# scp -r vast_tamperforge:/workspace/tamperforge/results/p1_cleanbase_rank* results/
# scp -r vast_tamperforge:/workspace/tamperforge/results/p1_cleanbase_reference results/
```

Already copied local: `p1_cleanbase_nojudge/`, `p1_cleanbase_nojudge_v2/`.

## 2. The rank sweep (THE decisive experiment)

Question: at the *smallest* k where ablating k adapter dirs removes safety
(ASR ~0.68 like bare), is the entangled-k capability much worse than random-k?
If yes -> entanglement real. If no window -> thesis needs rethink.

`git pull --ff-only` on vast first (uses new `--conditions` flag).

### 2a. Reference conditions, once (~7 min)

```bash
cd /workspace/tamperforge && git pull --ff-only
python experiments/p1_mad_crux.py \
  --adapter outputs/safety_adapter_p1_cleanbase.pt \
  --model-id google/gemma-3-1b-it --advbench-source walledai \
  --adapter-layer 13 --direction-layer 13 --abliterate-layers all \
  --n-direction 256 --n-advbench 50 --n-arc 100 \
  --max-new-tokens 128 --no-judge \
  --conditions base,base_ablated,base_adapter \
  --run-id p1_cleanbase_reference
```

### 2b. Sweep, k-dependent conditions only (~35 min)

```bash
for k in 1 2 4 8 16 32 64; do
python experiments/p1_mad_crux.py \
  --adapter outputs/safety_adapter_p1_cleanbase.pt \
  --model-id google/gemma-3-1b-it --advbench-source walledai \
  --adapter-layer 13 --direction-layer 13 --abliterate-layers all \
  --adapter-attack-rank $k \
  --n-direction 256 --n-advbench 50 --n-arc 100 \
  --max-new-tokens 128 --no-judge \
  --conditions base_adapter_ablated_full,base_ablated_randN \
  --run-id p1_cleanbase_rank$k ; done
```

Read per k: `base_adapter_ablated_full` (entangled top-k) vs `base_ablated_randN`
(random k). Win = smallest k with entangled ASR ~0.68 AND entangled ARC/PPL
wrecked while random-k ARC/PPL healthy.

## 3. LLM judging (LOCAL, env_ml — user runs, not the box)

Confirms keyword "ASR 1.0" on ablated conditions is fake (lobotomized model, not
harmful-compliant). Judge the copied p1 generations locally:

```bash
cd "/Users/aaronrockmenezes/Desktop/Projects/Mech Interp/tamperforge"
PY=/Users/aaronrockmenezes/miniforge3/envs/env_ml/bin/python

PYTHONPATH=src $PY experiments/judge_generations.py \
  --generations results/p1_cleanbase_nojudge_v2/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode --judge-max-tokens 512 --num-workers 14 \
  --run-id p1_cleanbase_v2_judged
```

Note: `generations.jsonl` has all 6 conditions tagged in each row's `condition`
field. `judge_generations.py` scores every row and reports one overall summary;
split by `condition` afterwards for per-condition ASR. Repeat for
`p1_ablbase_nojudge/generations.jsonl` once copied.

## 3b. FIX p1 for the abliterated-base (product) variant

The `p1_ablbase_nojudge` run is **invalid for hole #1**: `p1_mad_crux.py` loads
the clean `--model-id` for every condition, so the ablbase adapter sat on a base
that still had native refusal → `adapter_only` stayed at ASR 0 (native refusal
carried safety), not because the adapter held. Same direction-count confound as
cleanbase otherwise.

To test the product properly, p1 must strip the base's native refusal first for
the adapter conditions (matching how the ablbase adapter was trained). Options:
- add `--base-native-ablate` that runs `abliterate_model_inplace(model,
  empirical_refusal_dir, layers)` on the loaded base before applying the adapter,
  for `base_adapter*` conditions; or
- accept a `--base-checkpoint` pointing at
  `outputs/gemma3_1b_it_abliterated_all_empirical` and load that as the base for
  adapter conditions.
Then re-run the ablbase adapter through the (fixed) sweep.

## 4. Design levers if the sweep shows no window

- Retrain adapter with small `--d-hidden` (8-16) so safety lives in few dirs;
  re-test whether ablating those few entangled dirs costs more than few random.
- Raise `--lambda-entangle` to sharpen W_out<->language alignment per direction.
- Expand `BENIGN_PROMPTS` / add `--benign-source` (only 30 hardcoded now) for a
  broader suppression + KL probe set.

## 5. Housekeeping (deferred, parked until P1 verdict)

- Speed up `generate_responses`: pad-batch prompts instead of one-at-a-time
  (~5-10x). And precompute frozen benign base activations in `_suppress_loss`.
- p1 hardcodes nothing now, but confirm before generalizing to non-Gemma.
