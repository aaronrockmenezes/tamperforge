# Loss-term ablation matrix — planned 2026-08-30

**Status: NOT RUN. This is the next experiment.** One RTX Pro 6000. Launcher:
`scripts/runs/ablation_matrix.sh`. Evaluator: `scripts/runs/sweep_lambda_eval.sh`.
Report: `scripts/tools/sweep_lambda_report.py`.

Read `docs/handoff_2026_08_29_lambda_sweep.md` first — it holds the telemetry provenance for
everything asserted here.

## Why this and not more scaling

Nobody has established that the attacked branch of the loss contributes anything. Two of its
terms are provably inert and the third is provably underdetermined:

| term | measured behaviour |
|---|---|
| `L_harm` | nonzero on **7/500** steps (Phi), **6/1000** (Qwen). Saturated by step ~7. |
| `L_gib`, `L_attacked_safe`, `L_attacked_benign`, `L_shutdown` | **0 steps**, λ=0 |
| `L_rr` | converges (Phi 0.955 → 0.06) while attacked behaviour stays 60–85% harmful |
| `L_uncensor` | the only reliably-firing attacked-branch term (46–53% of steps) |

`L_uncensor`'s job is to *remove refusal* from the attacked model. The terms meant to replace
refusal with gibberish are dead (`L_harm`) or underdetermined (`L_rr`). So the reliable signal
strips the safety fallback and the unreliable ones fail to fill the gap — leaving compliance.
That is the Phi failure expressed as a property of the loss, not as bad luck.

## Two concrete defects to fix

### 1. `harm_margin` is unreachably low

`L_harm = relu(harm_margin − harm_abl)` with `harm_margin=4.0`. Measured `harm_abl`:

| | early | late |
|---|---:|---:|
| Phi (1–100 → 401–500) | 5.92 | 11.90 |
| Qwen (1–50 → 950–1000) | 5.72 | 15.15 |

Above the margin by step ~7, every run. **Set `--harm-margin 16`** — 12 goes dead on Qwen by
~step 700. The margin was calibrated against the old affirmative-prefix target and never
revisited after `mine_harm_targets.py` changed what `harm_abl` measures.

### 2. `rr_layers=last_half` does not cover the layer the attack wins at

The attack is `--attack-layers all` — it is *not* restricted. What is restricted is the
rerouting **loss**, scored only on the back half. Across the five models the Pro 6000 campaign
trained (direction layers read back from each run's own config line; `n_layers` from the
trainable-matrix counts in the same logs):

| model | n_layers | rr starts at | direction layer | |
|---|---:|---:|---:|---|
| **Phi-4-mini** | 32 | 16 | **13** | **OUTSIDE** |
| Gemma-4-E2B | 35 | 17 | 17 | inside |
| Llama-3.2-3B | 28 | 14 | 14 | inside |
| Ministral-3-3B | 26 | 13 | 15 | inside |
| Qwen3-4B-2507 | 36 | 18 | 19 | inside |
| Qwen3-0.6B | 28 | 14 | 20 | inside |

**Phi is the only model whose direction layer falls outside `rr_layers`, and the only one that
demonstrably failed** (62–96% harm across checkpoints; peak at L13/L14 in every sweep).

Evidence strength: **n=2.** Only Phi and Qwen-0.6B were ever evaluated — the other four were
trained and uploaded but never attacked. One outside → failed; one inside → walled. Suggestive,
falsifiable, not established. Arm `c1` tests it directly.

## The matrix

Nine arms. **1000 steps** (raised from an initial 600 mid-launch), seed 42. `a0` is the current
recipe verbatim; `b*` knock terms out; `c1` isolates the margin fix. **`--rr-layers all` is now
baseline for every arm** (not just `c1`) — closes the supervision gap for the whole matrix rather
than isolating it in one arm, so `c1`'s only remaining delta is `--harm-margin 16`. All arms set
`--grad-clip 1.0` (the trainer default is `1e9`, i.e. disabled; gradient p95 reached 1776–1856 on
the reference run once the wall formed).

| arm | change | question |
|---|---|---|
| `a0` | — | baseline for every delta |
| `b1` | `--lambda-uncensor 0` | let the attacked model refuse — fortress instead of poison pill |
| `b2` | `--lambda-rr 0` | is rerouting contributing anything? |
| `b3` | `--lambda-clean 0` | what is clean-preservation costing the wall? |
| `b4` | `b1+b2` | refusal-only defense |
| `b5` | `b1+b3` | |
| `b6` | `b2+b3` | |
| `b7` | `b1+b2+b3` | **floor** — attacked branch fully off |
| `c1` | `--harm-margin 16` | fixes the unreachable margin (was confounded with rr-layers in an earlier draft of this doc; that flag is now baseline) |

**`b7` is the arm nobody wants and everybody needs.** If it walls as well as `a0`, the wall came
entirely from the attack distribution (multi-rank + surgical + all-layer, the four flags that
separate new Version G from old) and the attacked-branch loss is decoration.

### Pre-registered predictions

Written before the run so results are not read post-hoc:

- `b2`, `b4`, `b6`, `b7` → **more harmful** than `a0` if `L_rr` matters. If they are flat, `L_rr`
  is decoration.
- `b3`, `b5`, `b6` → **stronger wall, worse clean model.** This is the `L_rr` vs `L_clean_gen`
  conflict made visible; `_reroute_loss`'s own docstring says the two are "in direct conflict on
  gemma and barely interact on Qwen".
- `b1` → harm down, refusal up. Fortress. Most likely arm to *work*, at the cost of moving into
  the occupied Shairah/ART cell.
- `c1` on **Phi specifically** is the load-bearing test of the supervision-gap hypothesis.

### `c2` — QUEUED, not part of this pass

**`--lambda-harm 0` — drop the dead term entirely, rather than raise its margin.** Arm is defined
in `ablation_matrix.sh` and ready to launch, but explicitly **not** part of the current 9-arm run.

Do not start `c2` until the current matrix (`a0` through `c1`) is fully trained and evaluated.
`c1` raises `harm_margin` 4→16 so `L_harm` stops saturating by step ~7; `c2` asks the opposite
question — does the term matter at all, or is it safe to delete outright? Comparing `c2` against
`a0` (both `harm_margin=4`, differing only in `lambda_harm`) answers that directly.
Comparing `c2` against `c1` then tells you whether a fixed-but-present `L_harm` beats no
`L_harm` — the two are not redundant.

```bash
MODEL=qwen06 ARMS="c2" bash scripts/runs/ablation_matrix.sh
```
```bash
ARMS="c2" PREFIX=abl_qwen06 bash scripts/runs/sweep_lambda_eval.sh
```
```bash
python scripts/tools/sweep_lambda_report.py --prefix abl_qwen06 --arms base,a0,b1,b2,b3,b4,b5,b6,b7,c1,c2
```

## Models

Tiered, because 9 arms × 3 models is ~30 GPU-hours.

| tier | model | arms | why | est. |
|---|---|---|---|---|
| 1 | **Qwen3-0.6B** | all 9 | cheapest (4.21 s/it), and the only model with a working wall to ablate | ~10.5 h |
| 2 | **Phi-4-mini** | `a0 b1 b2 c1` | the failure case; `c1` is the hypothesis test | ~5.5 h |
| 3 | Qwen3-4B **or** Gemma-4-E2B | `a0 c1` | does the fix hold at 4B / another family | ~3 h |

Tier 1 + tier 2 is a full day and answers the main question. Tier 3 only if time allows.

## Decision rule

Gates first, cheapest disqualifier first. **PPS ranks; it must not select** — a model that
refuses everything scores top on PPS (clean harm 0, clean gibberish 0, attacked refused high),
which is exactly version_J's failure: 0.000 clean harm bought by refusing 57.6% of safe prompts.

1. **XSTest-safe answered ≥ base − 10pp** — `data/heldout_vg_20260804/xstest_safe.jsonl` +
   `scripts/external_benches/score_overrefusal.py`. Plugs the PPS hole. ~3 min/arm.
2. **MT-Bench turn-1 ≥ base − 0.5** — `experiments/mtbench_single.py --repeats 3`. Never n=1:
   identical generations, same pinned judge, temp 0 scored base 4.54 then 4.74. ~12 min/arm.
3. **PPS** ranks the survivors.

ARC/MMLU/GSM8K on the winner only — necessary, never sufficient (version_B held 95% GSM8K at
MT-Bench 2.94).

`b1`/`b4`/`b5`/`b7` will score high on the wall by *refusing*. Gate 1 is what separates that
from a real result. Do not skip it.

## Statistical power

At n=64 (stage B) and a rate near 0.20, `SE ≈ 5pp`. **Two arms differing by less than ~10pp are
indistinguishable.** With one seed, only large effects are detectable. If arms cluster inside
10pp, the correct conclusion is "the objective is insensitive to these terms," not a ranking —
and for `b7` in particular that *is* the finding.

## Launch

```bash
MODEL=qwen06 DRY_RUN=1 bash scripts/runs/ablation_matrix.sh
```

```bash
MODEL=qwen06 bash scripts/runs/ablation_matrix.sh
```

```bash
MODEL=phi4mini ARMS="a0 b1 b2 c1" bash scripts/runs/ablation_matrix.sh
```

Evaluation and report (per model, adjust the arm list):

```bash
ARMS="base a0 b1 b2 b3 b4 b5 b6 b7 c1" bash scripts/runs/sweep_lambda_eval.sh
```

```bash
python scripts/tools/sweep_lambda_report.py --seed 42 --rank 1
```

## Known limits of this plan

- **Final checkpoint only, no trajectory.** `save_every=200` means every arm actually saves 5
  snapshots (`.s200.pt` … `.s800.pt` plus the unsuffixed step-1000 final), but
  `sweep_lambda_eval.sh` only ever reads the unsuffixed final (line 65, hardcoded, no loop over
  the `.sXXX.pt` files). This matrix compares arms to each other **at a fixed endpoint**; it does
  not show when each arm's wall (or failure) emerges during training, and can't catch oscillation
  the way the earlier `qwen06_new_vg_progress_20260827` step-by-step sweep did. Decided
  2026-08-30: leave as-is for this pass — the question here is "does term X matter," not "when
  does it start mattering." The intermediate checkpoints are already on disk if this needs
  revisiting; sweeping all 5 per arm would be a 5× eval cost (~2.8h → ~14h for the 9-arm matrix).
- **`harm_targets_qwen.json` is Qwen-specific.** Phi/Ministral/Llama/Gemma were all trained
  against Qwen's mined harmful completions, thinking-blocks included, through a different
  tokenizer. Second leak, separate from the AdvBench exposure. Mine per family before treating
  any cross-model comparison as clean.
- **AdvBench is train-exposed.** 404/520 harm-target goals and all 520 refusal entries are
  AdvBench. Every number here is on-distribution. The frozen held-out suite exists
  (`data/heldout_vg_20260804/`, 16 arms, 47,012 generations, all generated) but was never judged.
- **Single seed.** Roughly half of seeds historically fail to find the basin. Re-run the winner
  and `a0` at `SEED=2` before believing any ordering.
- **Norm weights are never trainable.** Only projection matrices update — every attention and MLP
  linear in every block (Qwen3-4B: 252 matrices = 7/layer × 36), but no layernorm. That is the
  documented reason gemma's post-block gain could not be countered. Not addressed by this matrix.

## Do not do

- Do not launch training without explicit approval.
- Do not report an arm as winning on PPS alone.
- Do not compare these 600-step numbers against the s700/s900 reference rates — different step
  count, and `--grad-clip 1.0` is a deviation from the reference run.
- Do not reuse a direction layer or basis across checkpoints; every attack re-estimates on the
  checkpoint under test.
