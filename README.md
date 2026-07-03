# tamperforge

> A pre-release procedure that entangles **safety** with **capability** in
> open-weight LLMs, so that cheap automated uncensoring is self-defeating.

Model makers run tamperforge on a finished model (after pre-train / SFT / RLHF,
**before** publishing weights). The released checkpoint then resists the cheap,
scripted attacks: abliterating safety also degrades capability — *smart-and-safe
or dumb-and-dangerous*.

This is the **framework** spinoff of the [mindweather](https://github.com/aaronrockmenezes/mindweather)
research repo (which holds the blog drafts + exploratory experiments).

## What it does / does not claim

- ✅ **Kill cheap abliteration** — abliterating the tamper-resistant model yields ~0%
  coherent harm / ~100% gibberish AND craters capability, while abliterating the base
  gives coherent harm with capability intact. Generalizes off-distribution + survives a
  per-layer adaptive attack + Heretic's KL-optimizer. MAD verified across gemma/Qwen/Llama.
- ✅ **ABL-v8 = the conditional wall (current product).** v8 fixes v7's clean tax: the
  clean model is **base-like** — safe, coherent, capable, AND helpful (v7 over-refused 98%
  of benign prompts; v8 = 0.31 ≈ base), while abliteration still self-destructs. Proven on
  Qwen + Llama (Llama's off-dist leak closed: HarmBench 0.42→0.000). Mechanism: generative
  clean-anchor + two-stage curriculum. gemma v8 = in progress.
- ⚠️ **Honest limits:** clean prefill hole (non-gradient); v8_att MAD crater is
  task-dependent (kills code/instructions, math softer); seed/snapshot selection needed.
- ❌ **Not** fine-tune-proof (FTR thread = closed negative; abliteration-resist ≠ FT-resist).

Read **`CLAUDE.md`** + the live handoff **`docs/handoff_2026_07_03_MASTER.md`** first,
then **`MEMORY.md`**, **`THREAT_MODEL.md`**, **`ROADMAP.md`**, and the latest results:
`docs/devlog_2026_07_04.md` (ABL-v8), `results/mx_summary/*` (full matrices),
`docs/findings_external_benches_ifeval_2026_07_03.md`.

## Status

**MAD thesis VERIFIED on gemma-3-1b (single seed).** Abliteration-resistance (ABL-v7)
is the strong, working result; the paper anchors here. FT-resistance is unsolved
(active). Eval harness (LLM judge + `usefulness_label`, prefill/HarmBench/BeaverTails,
lm_eval ARC/MMLU) all built and run at full datasets. Open for publication: multi-seed,
multi-model, adaptive attacker (OBLITERATUS), GSM8K, TamperBench/ART baselines.
Latest numbers: `docs/findings_prefill_harmbench_beavertails_2026_07_02.md`.

## Install

```bash
conda activate env_ml      # uses existing env — never scaffold a new venv
./setup.sh                 # editable install + HF-auth check + smoke test
```

## Layout

```
src/tamperforge/      # the library (source of truth)
  model.py            # load model, optional SAE, device (cuda/mps/cpu), residual capture
  abliterate.py       # weight-space abliteration (the attack)
  directions.py       # empirical refusal directions; SAE helpers for later analysis
  adapter.py          # prototype entanglement block
  safety.py           # keyword fallback only; use judge for publishable ASR
  data.py             # AdvBench loader (+ load_advbench_prompts: tuple-gotcha-safe)
  eval/               # safety gen, OpenRouter judge, ARC/MMLU/PPL, logging
experiments/          # thin CLIs: baseline, judge, ablation, training, P1
configs/              # model + experiment configs (no CLI flag soup)
data/                 # advbench csv, features_safety.json
results/              # JSON outputs
outputs/              # local checkpoints (gitignored model weights)
docs/results_*.md     # human-readable result snapshots
docs/common_issues.md # known infra/eval failures and fixes
```

## Core facts

- Model: `google/gemma-3-1b-it` (26 layers, d_model=1152, bf16).
- P1 proof-of-concept does not depend on SAEs. SAE support is retained for later
  mechanistic analysis.
- bf16 → fp32 cast before any numpy / small-vector matmul.
- Future ablation experiments use all layers by default. L13-only is a legacy
  comparison.

## References

- Arditi et al. 2024 — Refusal is Mediated by a Single Direction (the attack).
- Tamirisa et al. 2024 — TAR: Tamper-Resistant Safeguards (closest prior work).
- Rosati et al. 2024 — RepNoise.
- Gemma Scope 2 (DeepMind), SAE Lens (jbloomAUS).
