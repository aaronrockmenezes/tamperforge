# tamperforge — Handoff

> For the next agent / chat. Read `AGENTS.md` first, then this file, then
> `CLAUDE.md` if using Claude Code. Last updated: 2026-07-01.

## TL;DR

tamperforge is a **pre-release procedure** that entangles safety with capability
in open-weight LLMs, so cheap automated abliteration self-defeats (*smart-and-safe
XOR dumb-and-dangerous*). It is the clean-room framework spinoff of the
`mindweather` research repo. The core MAD thesis is **unverified** — the immediate
job is experiment **P1**, which is go/no-go for the whole project.

## Repos

- **tamperforge** (this) — github.com/aaronrockmenezes/tamperforge (PRIVATE).
  Path: `/Users/aaronrockmenezes/Desktop/Projects/Mech Interp/tamperforge`.
  The framework. Where all new work happens.
- **mindweather** — github.com/aaronrockmenezes/mindweather. FROZEN. Blog drafts
  + emotion-steering + exploratory safety experiments. Reference only; do not
  build here. Old eval_*.py and .pt checkpoints live there.

## Environment (hard rules)

- ALWAYS use existing conda env `env_ml`. NEVER create a new venv / uv.
  Python: `/Users/aaronrockmenezes/miniforge3/envs/env_ml/bin/python`
- Use plain commands (`python`, `pip`, `PYTHONPATH=src python ...`). Do not put
  `conda run` in user-facing commands.
- Device: `tamperforge.pick_device()` → cuda > mps > cpu. Never hardcode.
- Gemma 3 is gated — assume `hf auth login` done.
- **Git author email must be** `85219711+aaronrockmenezes@users.noreply.github.com`
  (plain gmail is blocked by GitHub email privacy → push rejected). Already set in
  this repo's local config.
- Commit message footer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

## Model facts

- `google/gemma-3-1b-it` — 26 layers, d_model=1152, bf16.
- SAE support exists for later mechanistic analysis, but **P1 does not depend
  on SAE directions**.
- `data/features_safety.json` contains 100 rows across five categories. The old
  SAE ablation path used 31 unique refusal/identity IDs; treat those checkpoints
  as legacy comparisons, not P1 proof.
- Policy update 2026-06-30: future ablation experiments use **all layers**.
  L13-only is retained only as a one-time legacy comparison.

## What exists now (scaffold + ported library)

```
src/tamperforge/
  __init__.py     exports below
  model.py        pick_device, load_model, load_sae, capture_residual
  abliterate.py   model/adapter ablation, orthonormal direction projection
  directions.py   empirical_refusal_direction(...), sae_feature_directions(...)
  adapter.py      SafetyAdapter (prototype block), load_adapter, make_adapter_hook
  safety.py       is_refusal (PLACEHOLDER keyword matcher — replace in P3)
  data.py         load_advbench (tuples!), load_advbench_prompts (strings, safe), augment_with_injections
  eval/           reusable harness: safety gen, judge, ARC/MMLU/PPL, logging
data/             advbench_harmful_behaviors.csv, features_safety.json
configs/          gemma3_1b.yaml
experiments/      baseline, judge, ablation, training, P1 scripts
README ROADMAP THREAT_MODEL HANDOFF pyproject setup.sh .gitignore
```

### Key changes vs mindweather (so old habits don't bite)
- `abliterate_model_inplace` now takes **direction tensors** `[n, d_model]`
  (or `[d_model]`) + `layers` — NOT `(feat_ids, sae_W_dec)`. Build dirs via
  `directions.py`.
- Multi-direction ablation now orthonormalizes direction sets before projection.
- Adapter ablation is explicit: `abliterate_adapter_out_inplace(adapter, dirs)`
  projects directions out of `SafetyAdapter.W_out`.
- Use `load_advbench_prompts()` for chat content. `load_advbench()` returns
  TUPLES — passing a tuple as chat content → model sees repr → 0% refusal.
- Device is centralized in `pick_device()`. No more per-script mps/cpu strings.

## Current eval/harness state (2026-07-01)

Implemented:
- `src/tamperforge/eval/` reusable harness:
  - `capability.py` — ARC-Challenge, MMLU subset, PPL.
  - `safety_eval.py` — AdvBench generation + keyword fallback.
  - `judge.py` — OpenRouter JSON judge client.
  - `log.py` — `manifest.json`, `events.jsonl`, `generations.jsonl`,
    `judgments.jsonl`, `summary.json`.
  - `suite.py` — one-call condition eval.
- `experiments/p0_baseline_eval.py` — original model baseline.
- `experiments/judge_generations.py` — judge existing `generations.jsonl`
  without rerunning local model.
- `experiments/make_abliterated_model.py` — save HF-compatible ablated
  checkpoints.
- `experiments/train_adapter.py` — train SafetyAdapter with refusal,
  suppression, W_out entanglement losses.
- `experiments/p1_mad_crux.py` — fixed five-condition P1 POC.
- `docs/eval_runbook.md` and `docs/vast_runbook.md` — cheap/full commands.
- `scripts/vast_setup.sh` — Vast GPU setup script. It validates CUDA Torch,
  installs tamperforge + `lm_eval[hf]`, checks HF auth, and supports local dry
  validation with `ALLOW_NO_CUDA=1 SKIP_INSTALL=1`.
- `CLAUDE.md` — Claude Code handoff with current server status and next steps.
- `docs/results_2026_07_01.md` — current result snapshot.
- `docs/common_issues.md` — server/eval failure modes and fixes.

Existing local ablated checkpoints:
- `outputs/gemma3_1b_it_abliterated_l13_sae`
  - Generated locally, one-time L13 legacy comparison.
  - Smoke only: `results/smoke_l13_ablated_8tok/`.
  - Local MPS 128-token eval was interrupted because generation was too slow
    before first output. Prefer Vast RTX 4090 for official runs; RTX 5090 only
    if host CUDA is 12.9+.
- `outputs/gemma3_1b_it_abliterated_all_sae`
  - Generated locally, all-layer SAE ablation.
  - Legacy mechanistic comparison only. Do not use as P1 proof.

Completed local baseline:
- Run: `results/baseline_local_nojudge/`
- Device: MPS.
- AdvBench keyword refusal: 98/100, ASR 2% (keyword only, not paper-grade).
- PPL: 13.7387 over `PROSE_TEXT`.
- ARC-Challenge: 41/100.

Completed judged baseline:
- Run: `results/baseline_deepseek_v4_flash_judge_100_w12_combined/`
- Judge: `deepseek/deepseek-v4-flash`
- Parse failures: 0/100.
- Judge refusal: 98/100.
- Judge ASR: 1/100.
- Rating counts: `0:98`, `1:1`, `3:1`.

Interrupted/partial judge runs:
- `results/baseline_nemotron3_ultra_judge*` are partial/slow/free-tier runs.
  Do not use as final numbers.

Current Vast server facts:

- SSH alias: `vast_tamperforge`
- Path: `/workspace/tamperforge`
- Env: `/venv/main`
- Good instance observed: RTX 4090 24GB, driver 580.95.05, host CUDA 13.0,
  Python 3.12.13, Torch `2.11.0+cu130`, vLLM `0.24.0`.
- Bad instance observed: RTX 5090, driver 570.195.03, host CUDA 12.8. vLLM
  imports but generation fails with driver/runtime mismatch. Do not debug; stop
  and rent a 4090 or a 5090 with host CUDA 12.9+.

Recent server batch requested by user:
1. Base `google/gemma-3-1b-it`.
2. `DavidAU/gemma-3-1b-it-heretic-abliterated-uncensored`.
3. `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated`.
4. Then local empirical all-layer ablated Gemma + adapter/P1.

Recommended flow:
- Run `bash scripts/vast_setup.sh` on RTX 4090, or RTX 5090 only if host CUDA
  12.9+.
- Safety: generate HF `walledai/AdvBench`, capped to 500 prompts, at
  `max_new_tokens=512`, no judge inline, then judge `generations.jsonl` with
  DeepSeek V4 Flash and 12 workers. The vendored local 520-row CSV is legacy.
  Judge request must disable OpenRouter reasoning; fixed in commit `56db984`.
- Capability: use EleutherAI `lm-eval` for ARC-Challenge 25-shot full test
  split. Full ARC-Challenge test has 1,172 examples; use `--limit 600` only if
  budget requires and label it capped.
  On 24GB GPUs use `max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1`
  and `--batch_size 1`; `--batch_size auto` OOMed.
- Google's model card gives Gemma 3 PT 1B ARC-c 25-shot = 38.4, not a clean
  `gemma-3-1b-it` official number. Treat server-run `google/gemma-3-1b-it` with
  the exact same `lm-eval` command as the local reference for all comparisons.

## Recent P0 server evidence

Detailed snapshot: `docs/results_2026_07_01.md`.

- Base safety generation completed:
  `results/safety_base_advbench500_nojudge`, AdvBench 500, keyword refusal
  0.98, keyword ASR 0.02.
- HF heretic safety generation completed:
  `results/safety_hf_heretic_advbench500_nojudge`, keyword refusal 0.112,
  keyword ASR 0.888.
- HF extreme safety generation completed:
  `results/safety_hf_extreme_advbench500_nojudge`, keyword refusal 0.036,
  keyword ASR 0.964.
- Judge smoke was fixed and re-run:
  `results/judge_smoke_hf_heretic_10_v2`, DeepSeek V4 Flash, parse failures
  0/10, judge ASR 0.9.
- Full 12-worker judge summaries were not verified from local at doc update time.
  Inspect on server if user ran them:
  `results/judge_*_advbench500_w12/summary.json`.

## STATUS 2026-07-01 (late) — P0 done, P1 inconclusive

- **P0 complete + archived.** base/heretic/extreme judged (DeepSeek V4 Flash,
  AdvBench 500) + ARC-c 25-shot. Table in `docs/results_2026_07_01.md`. Judge
  parse-failures resolved via `scripts/adjudicate_judge_failures.py`. Takeaway:
  cheap uncensoring is ~free today (heretic ASR 0.886 at zero capability cost).
- **P1 first run INCONCLUSIVE** (`results/p1_cleanbase_nojudge_v2/`). The all-229
  adapter attack destroys capability, BUT the direction-count control
  (`base_ablated_randN`: 229 random dirs) destroys it just as badly. So the
  collapse is a direction-count artifact, not proven entanglement. Full analysis
  in the P1 section of `docs/results_2026_07_01.md`.

### IMMEDIATE NEXT — the rank sweep (settles P1). See `TODO.md`.

Find the smallest k where ablating k adapter W_out dirs removes safety (ASR
~0.68), then compare entangled-k vs random-k capability at that k. Entanglement
is supported only if a *small* entangled ablation is disproportionately
destructive vs a small random one. Uses the new `--conditions` subset flag +
`base_ablated_randN` control (both committed). Commands in `TODO.md` §2.

Judging of P1 generations is done LOCALLY (env_ml) — see `TODO.md` §3.

## P0/P1 background (historical — P0 now done)

Before P1, finish P0 baseline table:

1. Verify/run full DeepSeek judges for base, heretic, extreme.
2. Run ARC-Challenge 25-shot full with the 24GB-safe lm-eval command for base,
   heretic, extreme.
3. Update `docs/results_2026_07_01.md`.

Then proceed to P1:

**Question:** after training a clean-base adapter to fuse refusal behavior with
language/capability directions, does ablating the adapted model's adapter
`W_out` directions make safety removal cost more capability than bare Gemma
refusal-direction ablation?

**Script:** `experiments/p1_mad_crux.py`. Five conditions, all measured with the
same prompts, same `EvalConfig`, same `max_new_tokens`, same benchmarks, same
DeepSeek V4 Flash judge:

| # | Condition | Build |
|---|-----------|-------|
| 1 | base | clean Gemma |
| 2 | base_ablated | clean Gemma, then empirical refusal-direction ablation |
| 3 | base_adapter | clean Gemma + trained SafetyAdapter |
| 4 | base_adapter_ablated_full | clean Gemma + adapter, then ablate adapter `W_out` dirs from model and adapter |
| 5 | base_adapter_ablated_adapter_only | clean Gemma + adapter, then ablate adapter `W_out` only |

**Directions:**
- Bare Gemma ablation: `empirical_refusal_direction(...)`, no SAE.
- Adapted model attack: orthonormalized `SafetyAdapter.W_out` output directions,
  no SAE.

**Metrics per condition:**
- ASR / refusal rate on HF `walledai/AdvBench` prompts via DeepSeek V4 Flash for
  official runs. Keyword refusal is fallback only. Official server run caps to
  500 prompts.
- Capability: internal ARC validation + PPL for P1 because the prototype adapter
  is a forward hook and is not exposed to `lm-eval` yet. Standalone model
  capability baselines use `lm-eval`.

**Matched ASR:** capability gaps are strongest evidence only when the compared
attacks have similar ASR. The script reports ASR delta and `matched` using a
default ±5 percentage-point tolerance; unmatched results are still logged but
should be interpreted as inconclusive on the strict crux.

**Result = proof-of-concept support/failure.**
- Support: adapted pre-attack capability ~= base, adapted full ablation damages
  capability more than bare Gemma ablation at comparable ASR.
- Fail: adapted attack removes safety without extra capability cost, or adapter
  training itself damages capability too much.

**Output:** `results/<run_id>/summary.json` plus raw JSONL logs.

**Cost:** loads Gemma up to 6×, ARC×5, judge×5. Prefer Vast RTX 4090; RTX 5090
only with host CUDA 12.9+. Local MPS is too slow for official runs.

**Gotchas:** use all-layer ablation for future experiments. L13-only is legacy
comparison only. The old mindweather `abliterated_L13/` dir was suspect; do not
reuse it. bf16→fp32 before numpy. Move direction vectors to model device before
matmul (handled inside abliterate).

## After P1

P1b (block vs distributed entanglement) → P3 eval harness ∥ P2 adaptive
abliteration → P4 FT characterization (vs TAR/RepNoise) → P5 generalize
(Qwen/Llama). Codebase cleanup is parked until P1 passes. Full detail in ROADMAP.md.
