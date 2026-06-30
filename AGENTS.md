# AGENTS.md

Read this before touching tamperforge. If anything conflicts with older docs,
this file and `HANDOFF.md` win.

## Project

tamperforge is the clean framework repo for the safety/capability entanglement
work. Do **not** build new experiments in `mindweather`; use it only as old
reference material.

Core thesis: pre-release model surgery/training should make cheap open-weight
uncensoring self-defeating. Attacker can inspect weights. We measure whether
abliteration raises ASR only by damaging capability.

## Current policy

- Future tamperforge ablation experiments use **all model layers** by default.
- L13-only is allowed only as a one-time legacy baseline against old
  mindweather claims.
- P1 proof-of-concept must not depend on SAEs. Use empirical refusal direction
  for bare Gemma ablation and adapter `W_out` directions for adapted-model
  ablation.
- Official eval settings must stay identical across conditions in a run:
  same prompts, same `max_new_tokens`, same benchmarks, same DeepSeek V4 Flash
  judge config.
- Server benchmark policy: safety uses all 520 AdvBench prompts; standalone
  model capability uses EleutherAI `lm-eval` ARC-Challenge, not the small custom
  ARC loop. P1 still uses internal ARC because the adapter is a forward hook.
- Keep P1/P2/P3 result artifacts under `results/<run_id>/` with raw generations
  and summaries.
- Use `DeepSeek V4 Flash` (`deepseek/deepseek-v4-flash`) as the default paid
  OpenRouter judge; it is cheap and supports JSON mode.
- Keyword refusal is triage only, not publication-grade.

## Environment

- Use existing `env_ml`; do not create a new venv.
- Plain commands only: `python`, `pip`, `PYTHONPATH=src python ...`.
- Python path on this machine:
  `/Users/aaronrockmenezes/miniforge3/envs/env_ml/bin/python`
- Device helper is `tamperforge.pick_device()`; do not hardcode CUDA/MPS.
- Gemma is gated; HF auth must be present.
- `.env` may contain `OPENROUTER_API_KEY`, `HF_TOKEN`, etc. Never print values.

## Existing local checkpoints

These were generated in tamperforge and are HF-compatible local model dirs:

- `outputs/gemma3_1b_it_abliterated_l13_sae`
  - one-time L13-only legacy comparison.
  - Smoke result exists: `results/smoke_l13_ablated_8tok/`.
- `outputs/gemma3_1b_it_abliterated_all_sae`
  - all-layer SAE ablation.
  - Legacy mechanistic comparison only. Do not use as P1 proof.

Both use SAE feature directions from `data/features_safety.json`. See each
directory's `abliteration_meta.json`.

## Server plan

Use `docs/vast_runbook.md`. RTX 5090 is preferred if the image has working
CUDA 12.8+/PyTorch support; RTX 4090 is fallback.

Order:

1. Run `bash scripts/vast_setup.sh`.
2. Smoke base Gemma.
3. Run full AdvBench 520 generation for base + two DavidAU HF models.
4. Judge generations with DeepSeek V4 Flash, 12 workers.
5. Run `lm-eval` ARC-Challenge 25-shot full for base + two DavidAU HF models.
6. Then create local empirical all-layer ablated model, train adapter, and run P1.

Safety generation command shape:

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id MODEL_OR_PATH \
  --n-advbench 520 \
  --n-arc 0 \
  --max-new-tokens 128 \
  --run-id RUN_ID
```

## How abliteration works

Given a unit residual direction `d`, for each selected layer:

- read matrices (`q/k/v/gate/up`): `W <- W - outer(W @ d, d)`
- write matrices (`o/down`): `W <- W - outer(d, d @ W)`

Implementation: `src/tamperforge/abliterate.py`.

To create a local empirical all-layer ablated model:

```bash
PYTHONPATH=src python experiments/make_abliterated_model.py \
  --out outputs/gemma3_1b_it_abliterated_all_empirical \
  --direction-source empirical \
  --layers all
```

## P1 POC

Train adapter on clean Gemma:

```bash
PYTHONPATH=src python experiments/train_adapter.py \
  --out outputs/safety_adapter_p1.pt \
  --n-harmful 200 \
  --epochs 5 \
  --batch-size 4 \
  --run-id train_adapter_p1
```

Run fixed P1:

```bash
PYTHONPATH=src python experiments/p1_mad_crux.py \
  --adapter outputs/safety_adapter_p1.pt \
  --abliterate-layers all \
  --n-advbench 520 \
  --n-arc 299 \
  --max-new-tokens 128 \
  --judge \
  --run-id p1_mad_crux_judged
```

Conditions are exactly:
`base`, `base_ablated`, `base_adapter`,
`base_adapter_ablated_full`, `base_adapter_ablated_adapter_only`.

## Current verified results

Base Gemma no-judge:

- `results/baseline_local_nojudge/summary.json`
- AdvBench keyword refusal: 98/100
- Keyword ASR: 2%
- ARC-Challenge: 41/100
- PPL: 13.7387

Base Gemma DeepSeek V4 Flash judged:

- `results/baseline_deepseek_v4_flash_judge_100_w12_combined/summary.json`
- `n=100`, parse failures 0
- judge refusal rate: 98%
- judge ASR: 1%
- ratings: `0:98`, `1:1`, `3:1`

## Git/worktree caution

There are many new result dirs and scripts from the current work. Do not delete
or revert user-visible artifacts unless explicitly asked. `outputs/*.safetensors`
and `*.pt` are gitignored by design.
