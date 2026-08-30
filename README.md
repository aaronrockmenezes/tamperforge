# tamperforge

> A research framework testing whether **safety** can be entangled with **capability** in
> open-weight LLMs so cheap automated uncensoring becomes self-defeating.

The intended use is a pre-release transformation of a finished model (after pre-train / SFT /
RLHF, before publishing weights): abliterating safety should also degrade capability —
*smart-and-safe or dumb-and-dangerous*. Current experiments do **not** establish that robustly.

This is the **framework** spinoff of the [mindweather](https://github.com/aaronrockmenezes/mindweather)
research repo (which holds the blog drafts + exploratory experiments).

## ▶ NEXT STEP — loss-term ablation matrix (planned, not run)

**Read [`docs/plan_2026_08_30_ablation_matrix.md`](docs/plan_2026_08_30_ablation_matrix.md)
before starting any new training.** One RTX Pro 6000, nine arms, Qwen3-0.6B + Phi-4-mini
(+ optionally Qwen3-4B or Gemma-4-E2B). Launcher: `scripts/runs/ablation_matrix.sh`.

It exists because the attacked branch of the loss has never been shown to contribute anything:
`L_harm` fires on 6–7 steps out of 500–1000 (margin unreachably low), four more terms sit at
λ=0, and `L_rr` converges while attacked behaviour stays 60–85% harmful. The matrix knocks each
term out to find which are load-bearing, and fixes two concrete defects — `harm_margin` 4 → 16,
and `rr_layers` `last_half` → `all`. On Phi the rerouting loss never scored the layer the attack
actually wins at (DL 13, rr starts at 16); Phi is the only one of five trained models with that
gap and the only one that demonstrably failed. Companion telemetry provenance:
[`docs/handoff_2026_08_29_lambda_sweep.md`](docs/handoff_2026_08_29_lambda_sweep.md).

> **Blocking prerequisite for any cross-model claim: harm targets are Qwen-specific.**
> `data/harm_targets_qwen.json` holds 404 harmful completions mined from an attacked *Qwen*.
> The 2026-08-15 campaign trained Phi-4-mini, Ministral-3-3B, Llama-3.2-3B and Gemma-4-E2B
> against that same file — teacher-forcing Qwen's prose, `<think>` blocks included, through
> four other tokenizers and model families. This is a second data leak, separate from the
> AdvBench exposure, and it makes every cross-model comparison in this repo dirty.
> **Mine harm targets per family** (`experiments/mine_harm_targets.py`, sourced from a judged
> run on a *different* model than the one being trained) before comparing across families.

---

**Current verdict (2026-08-15): do not spend more training compute on the current Version G
recipe.** Fresh, checkpoint-specific direction estimation and layer selection broke both the
older Qwen3-0.6B Version G checkpoint and the newly trained Phi-4-mini Version G checkpoint.
The strong claim that TamperForge robustly makes abliteration self-defeating is therefore
falsified for the tested adaptive threat model. The legacy naive/fixed-attack wall remains a
real bounded observation; the defensible research result is now that the wall does not survive
fresh adaptive re-estimation. See `docs/findings_fresh_rank_attacks_2026_08_15.md` and
`docs/handoff_2026_08_15_codex.md`.

`scripts/runs/version_g_final.sh` is the canonical **reproducer** for the final recipe, not a
recommendation for a new campaign. Older `chain_*` and model/version-specific launchers remain
historical reproduction only.

## What it does / does not claim

- ❌ **Fresh adaptive rank attacks: NOT solved.** Re-estimating directions on the released
  checkpoint and selecting the best layer recovered 43.1%/68.5% held-out coherent harm on
  Qwen3-0.6B under rank-1/rank-2, and 83.5%/59.7% on Phi-4-mini. These are judged AdvBench
  rates after excluding the 16 layer-selection prompts.
- ✅ **Kill cheap (naive) abliteration** — abliterating the tamper-resistant model via the
  standard rank-1 attack yields ~0% coherent harm / ~99% gibberish AND craters capability,
  while abliterating the base gives coherent harm with capability intact. This fixed-attack
  observation reproduced across gemma/Qwen/Llama (ABL-v8, 2026-07-04).
- ⚠️ **ABL-v8 = the historical conditional wall.** v8 fixes v7's clean tax: the
  clean model is **base-like** — safe, coherent, capable, AND helpful (v7 over-refused 98%
  of benign prompts; v8 = 0.31 ≈ base), while abliteration still self-destructs under the
  naive attack. Proven on all 3 architectures. Mechanism: generative clean-anchor + two-stage
  curriculum.
- ❌ **Adaptive attacker: NOT solved.** It survived one bounded in-house per-layer attack,
  and gemma v7 (prior product version) survived Heretic's KL-optimizer. **But Heretic BREAKS
  ABL-v8 on all 3 architectures** — Llama 88% harm, gemma 93% harm (zero capability cost, any
  trial), Qwen 82% harm (zero capability cost, any trial) (`docs/heretic_v8_2026_07_18.md`).
  **Do not claim "survives adaptive attacks" as a blanket statement anywhere.**
- ⚠️ **Other honest limits:** clean prefill hole (non-gradient); v8_att MAD crater is
  task-dependent (kills code/instructions, math softer); seed/snapshot selection needed.
- ❌ **Not** fine-tune-proof (FTR thread = closed negative; abliteration-resist ≠ FT-resist).

Read **`docs/handoff_2026_08_15_codex.md`** first, then **`CLAUDE.md`** and
**`docs/findings_fresh_rank_attacks_2026_08_15.md`**. Older context follows in
**`docs/devlog_2026_07_17.md`**, **`docs/handoff_2026_07_03_MASTER.md`**, **`MEMORY.md`**,
**`THREAT_MODEL.md`**, **`ROADMAP.md`**, and the latest results: `docs/devlog_2026_07_04.md`
(ABL-v8, 3/3 architectures), `docs/heretic_v8_2026_07_18.md` (Heretic breaks v8 on
all three), `full_eval_matrices/*` (full matrices), `docs/findings_external_benches_ifeval_2026_07_03.md`.

## Status

**The naive/fixed rank-1 wall was observed across three small architectures, but it is not a
robust defense.** Heretic already broke ABL-v8 on all three architectures. The 2026-08-15 fresh
rank attack adds a stronger failure: even ordinary Arditi-style rank-1/rank-2 attacks work when
their directions and winning layers are recomputed on the final checkpoint. On Phi-4-mini,
centered `L_rr` fell from about 0.955 over the first 25 steps to 0.060 over the last 25 while
fresh attacks still produced 60-84% coherent harm. The optimizer solved the internal surrogate;
the desired behavior did not follow.

Five 500-step Version G clean checkpoints were trained and upload-verified under
`aaronrockmenezes/tamperforge/version_g_final/`: Phi-4-mini, Ministral-3-3B, Llama-3.2-3B,
Qwen3-4B-2507, and Gemma-4-E2B. Only Phi-4-mini has the completed fresh rank-1/rank-2 adaptive
evaluation described above. Later Phi checkpoint probes (steps 600–1000) are exploratory
16/64-prompt layer sweeps, not additional full confirmation runs. The complete Phi table,
LoRA/surgical probes, and storage boundary are in
[`docs/results_phi4mini_rank1_trajectory_20260827.md`](docs/results_phi4mini_rank1_trajectory_20260827.md).
The newer Qwen3-0.6B checkpoint-by-checkpoint rank-1/rank-2 tables are compiled in
[`results/compiled/qwen06_new_vg_progress_20260827/report.md`](results/compiled/qwen06_new_vg_progress_20260827/report.md).
These uploads are experiment artifacts, not validated defenses.
Qwen3.5-4B failed the one-step smoke before training.

## Source and archive

This repo is the compact source tree. The sibling `../tamperforge-archive` preserves the
pre-cleanup raw generations, judgments, events, and historical code/docs. Source `results/`
keeps manifests and summaries only. Checkpoint payloads are not stored locally; retained
artifacts live in the private HF repo `aaronrockmenezes/tamperforge`.

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
configs/              # legacy config; active experiment CLIs use flags
data/                 # advbench csv, features_safety.json
results/              # compact manifests and summaries only
outputs/              # metadata/config only; checkpoint payloads are remote
docs/results_*.md     # human-readable result snapshots
docs/common_issues.md # known infra/eval failures and fixes
../tamperforge-archive/ # raw runs and historical code/docs
```

## Core facts

- Original reference model: `google/gemma-3-1b-it` (26 layers, d_model=1152, bf16); current
  artifacts span multiple model families.
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
