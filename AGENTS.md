# AGENTS.md

> **CURRENT STATE (2026-07-18): read `docs/devlog_2026_07_17.md` + `MEMORY.md`
> first.** The text below is durable project context (thesis, threat model,
> structure). Live status — v8 = 3/3 architectures against naive rank-1, but Heretic
> (adaptive attack) breaks the wall on all three, TamperBench validation in progress, Qwen3-8B scale
> attempt parked — is in the devlog + `CLAUDE.md`; those win on conflict.
> Source is intentionally compact: raw run artifacts and historical code/docs live in the
> sibling `../tamperforge-archive`; local checkpoint payloads were removed from both trees.

Read this before touching tamperforge. If anything conflicts with older docs,
this file, `CLAUDE.md`, and `docs/handoff_2026_07_03_MASTER.md` win.

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
- Server benchmark policy: safety uses HF `walledai/AdvBench`, capped to 500
  prompts for official runs; standalone model evals use vLLM. Capability uses
  EleutherAI `lm-eval --model vllm` ARC-Challenge, not the small custom ARC
  loop. P1 still uses internal Transformers loops because the adapter is a
  forward hook.
- Keep only `manifest.json` and `summary.json` under source `results/<run_id>/`.
  Raw generations, judgments, and events belong in `../tamperforge-archive/results/`.
- Use `DeepSeek V4 Flash` (`deepseek/deepseek-v4-flash`) as the default paid
  OpenRouter judge. The request must disable OpenRouter reasoning; see
  `src/tamperforge/eval/judge.py` and `docs/common_issues.md`.
- Keyword refusal is triage only, not publication-grade.

## Environment

- Local Mac: use existing `env_ml`; do not create a new venv.
- Plain commands only: `python`, `pip`, `PYTHONPATH=src python ...`.
- Python path on this machine:
  `/Users/aaronrockmenezes/miniforge3/envs/env_ml/bin/python`
- Device helper is `tamperforge.pick_device()`; do not hardcode CUDA/MPS.
- Gemma is gated; HF auth must be present.
- `.env` may contain `OPENROUTER_API_KEY`, `HF_TOKEN`, etc. Never print values.

Vast server:

- Use active env `/venv/main`.
- Current good box observed: RTX 4090 24GB, driver 580.95.05, host CUDA 13.0,
  Python 3.12.13, Torch `2.11.0+cu130`, vLLM `0.24.0`.
- RTX 5090 is acceptable only if `nvidia-smi` reports host CUDA 12.9+.
  RTX 5090 + host CUDA 12.8 is a known bad vLLM/FlashAttention combo.

## Checkpoint storage

There are no local `.pt`, `.bin`, or `.safetensors` payloads in either source or
`../tamperforge-archive`. Use the private HF repo `aaronrockmenezes/tamperforge` for
retained artifacts. Do not run `scripts/tools/push_to_hf.py`; its artifact list and model card
are stale. `outputs/` is metadata/config only unless a new local run explicitly creates
a temporary checkpoint.

## Server plan

Use `docs/vast_runbook.md`. RTX 4090 is the safest cheap target. RTX 5090 is
only acceptable with host CUDA 12.9+.

Order:

1. Run `bash scripts/setup/vast_setup.sh`.
2. Smoke base Gemma.
3. Run full walledai/AdvBench 500 generation for base + two DavidAU HF models.
4. Judge generations with DeepSeek V4 Flash, 12 workers.
5. Run `lm-eval` ARC-Challenge 25-shot full for base + two DavidAU HF models.
6. Then create local empirical all-layer ablated model, train adapter, and run P1.

Historical result summary: `../tamperforge-archive/docs/archive/results_2026_07_01.md`.
Common server/eval failure fixes: `docs/common_issues.md`.

Safety generation command shape:

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id MODEL_OR_PATH \
  --backend vllm \
  --advbench-source walledai \
  --n-advbench 500 \
  --n-arc 0 \
  --max-new-tokens 512 \
  --max-length 4096 \
  --vllm-batch-size 64 \
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
  --advbench-source walledai \
  --n-advbench 500 \
  --n-arc 299 \
  --max-new-tokens 512 \
  --max-length 4096 \
  --judge \
  --run-id p1_mad_crux_judged
```

Conditions are (6, after the direction-count control was added):
`base`, `base_ablated`, `base_ablated_randN`, `base_adapter`,
`base_adapter_ablated_full`, `base_adapter_ablated_adapter_only`.
Use `--conditions <comma list>` to run a subset (needed for rank sweeps),
`--adapter-attack-rank k` to ablate only the top-k adapter W_out directions.

**P1 status 2026-07-01: first run INCONCLUSIVE.** The all-directions adapter
attack destroys capability, but so does ablating the same number of *random*
directions (`base_ablated_randN`) — a direction-count confound. The open
question is the rank sweep (smallest k that removes safety; entangled-k vs
random-k capability). See `TODO.md` and the P1 section of
`../tamperforge-archive/docs/archive/results_2026_07_01.md`. Do not claim P1 pass/fail until the sweep runs.

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

P0 baseline COMPLETE (AdvBench 500 judged + ARC-c 25-shot full, DeepSeek V4
Flash, post-adjudication parse_failures=0):

| Model | ARC acc | acc_norm | judge refusal | judge ASR |
|---|---:|---:|---:|---:|
| base | 0.366 | 0.4002 | 0.982 | 0.014 |
| heretic | 0.3788 | 0.4130 | 0.102 | 0.886 |
| extreme | 0.3447 | 0.3746 | 0.082 | 0.896 |

Runs: `results/judge_{base,hf_heretic,hf_extreme}_advbench500_w14/`. Full table +
P1 analysis in `../tamperforge-archive/docs/archive/results_2026_07_01.md`.

## Git/worktree caution

The archive/cleanup worktree is intentionally large and uncommitted. Do not restore
archived raw results or historical trees into source. Do not edit or remove the retained
v7 trainer/launcher files without explicit instruction.
