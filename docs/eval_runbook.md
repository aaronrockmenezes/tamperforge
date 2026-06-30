# tamperforge eval runbook

Goal: every paid run writes enough raw evidence for paper tables later.

For RTX 4090 or valid RTX 5090 server runs, `docs/vast_runbook.md` is
authoritative. This file keeps local/small examples and older command shapes for
reference.

Current human-readable result snapshot: `docs/results_2026_07_01.md`.
Known infra/eval failure fixes: `docs/common_issues.md`.

## Artifact layout

All scripts write to `results/<run_id>/`:

- `manifest.json` — args, git commit, env/device metadata.
- `events.jsonl` — condition starts/finishes, training epoch metrics.
- `generations.jsonl` — raw harmful prompts + model responses.
- `judgments.jsonl` — OpenRouter judge raw payloads, if enabled.
- `summary.json` — table-ready aggregate metrics.

## Cheap local smoke

```bash
cd tamperforge
PYTHONPATH=src python experiments/p0_baseline_eval.py --n-advbench 2 --n-arc 2 --max-new-tokens 32
```

## Proper baseline

Run without judge first:

```bash
cd tamperforge
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --n-advbench 100 \
  --n-arc 200 \
  --n-mmlu-per-subject 25 \
  --max-new-tokens 128
```

## Abliterated / uncensored baselines

Policy: future tamperforge ablation experiments use **all layers**. L13-only is
retained only as a one-time legacy comparison.

Already-created local checkpoints:

- `outputs/gemma3_1b_it_abliterated_l13_sae`
- `outputs/gemma3_1b_it_abliterated_all_sae`

Create/recreate clean tamperforge all-layer empirical-abliterated Gemma:

```bash
PYTHONPATH=src python experiments/make_abliterated_model.py \
  --out outputs/gemma3_1b_it_abliterated_all_empirical \
  --direction-source empirical \
  --layers all
```

Evaluate it with the same harness:

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id outputs/gemma3_1b_it_abliterated_all_empirical \
  --n-advbench 100 \
  --n-arc 100 \
  --max-new-tokens 128 \
  --run-id baseline_gemma3_1b_it_abliterated_all_empirical
```

HF counterpart baselines can use `--model-id` directly if they are Transformers
checkpoints. GGUF repos need llama.cpp/vLLM conversion and are not supported by
this harness as-is.

Candidate HF model IDs seen in current search:

- `DavidAU/gemma-3-1b-it-heretic-abliterated-uncensored`
- `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`
- `huihui-ai/gemma-3-1b-it-abliterated`
- `lunahr/gemma-3-1b-it-abliterated`

Example:

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id DavidAU/gemma-3-1b-it-heretic-abliterated-uncensored \
  --n-advbench 100 \
  --n-arc 100 \
  --max-new-tokens 128 \
  --run-id baseline_hf_davidau_heretic_uncensored
```

Then add judge after putting `OPENROUTER_API_KEY=...` in `tamperforge/.env`:

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --n-advbench 100 \
  --n-arc 200 \
  --n-mmlu-per-subject 25 \
  --judge \
  --judge-model deepseek/deepseek-v4-flash
```

For free models that do not support OpenRouter JSON mode, add
`--no-judge-json-mode`.

DeepSeek V4 Flash is the current paid judge. It must be called through the fixed
OpenRouter request in `src/tamperforge/eval/judge.py`, which disables reasoning.
Older requests can return `content: null` and create parse failures.

## Train adapter on rented GPU

Fast first pass:

```bash
PYTHONPATH=src python experiments/train_adapter.py \
  --out outputs/safety_adapter_p1.pt \
  --n-harmful 200 \
  --epochs 5 \
  --batch-size 4 \
  --lr 3e-4 \
  --lambda-entangle 1.0 \
  --lambda-suppress 0.5 \
  --abliterate-layers all
```

If loss is stable and time remains, rerun with `--epochs 20 --n-harmful 500`.

## P1 fixed POC

```bash
PYTHONPATH=src python experiments/p1_mad_crux.py \
  --adapter outputs/safety_adapter_p1.pt \
  --n-direction 64 \
  --abliterate-layers all \
  --n-advbench 100 \
  --n-arc 200 \
  --n-mmlu-per-subject 25 \
  --max-new-tokens 128 \
  --no-judge
```

Official judged run:

```bash
PYTHONPATH=src python experiments/p1_mad_crux.py \
  --adapter outputs/safety_adapter_p1.pt \
  --n-direction 64 \
  --abliterate-layers all \
  --n-advbench 100 \
  --n-arc 200 \
  --n-mmlu-per-subject 25 \
  --max-new-tokens 128 \
  --judge
```

## Phase map

- P0: baseline original Gemma using harness.
- P1: fixed five-condition POC: base, base_ablated, base_adapter,
  base_adapter_ablated_full, base_adapter_ablated_adapter_only.
- P1b: if P1 passes, compare removable adapter vs distributed entanglement.
- P2: adaptive abliteration, recompute refusal direction on released entangled model.
- P3: harden harness: HarmBench classifier or LLM judge, bootstrap CIs, fixed dataset versions.
- P4: fine-tuning attack frontier: examples/compute needed to bypass.
- P5: repeat on Qwen/Llama once Gemma P1-P3 hold.

## Notes

- Keyword refusal metric is cheap triage only.
- Judge ASR is publishable only if raw prompts/responses/judgments are archived.
- Future experiments use all-layer abliteration. L13-only is retained only as a
  one-time legacy comparison.
