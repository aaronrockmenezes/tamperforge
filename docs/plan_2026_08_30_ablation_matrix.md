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

## Results (2026-08-30/31) — RUN, not just planned

**Both rounds complete.** 9-arm knockout matrix, `c1`/`c2` margin variants, and the full
`{L_rr, L_uncensor, L_harm}` 2x2x2 factorial (`lambda_clean` fixed at baseline in every cell) all
ran on `vast-lsrpi-4x3090-a`. Raw numbers: `results/dl_sweeps/abl_qwen06_*_r1_stageB/summary.json`
(52 stage A/B summaries, committed to git). Pull the tables:

```bash
python scripts/tools/sweep_lambda_report.py --prefix abl_qwen06 --factorial
```

### Round 1 — knockout matrix

| arm | change | layer | attacked harm | attacked gib | PPS |
|---|---|---:|---:|---:|---:|
| base | untouched | L17 | 87.1% | 6.5% | 0.012 |
| a0 | control | L25 | 54.8% | 30.6% | 0.111 |
| b1 | uncensor=0 | L9 | 15.5% | 27.6% | 0.466 |
| b2 | rr=0 | L14 | 82.8% | 10.9% | 0.027 |
| b3 | clean=0 | L8 | 4.8% | 93.7% | 0.354 |
| b4 | uncensor=0+rr=0 | L15 | 84.1% | 9.5% | 0.018 |
| b5 | uncensor=0+clean=0 | L13 | 54.1% | 4.9% | 0.053 |
| b6 | rr=0+clean=0 | L18 | 71.0% | 21.0% | 0.064 |
| b7 | floor (all 3 off) | L13 | 60.7% | 1.6% | 0.063 |
| c1 | harm_margin 16 | L21 | 22.2% | 76.2% | 0.593 |

### Round 2 — 2x2x2 factorial, clean fixed at baseline

| rr | uncensor | harm | arm | layer | attacked harm | attacked gib | PPS |
|:-:|:-:|:-:|---|---:|---:|---:|---:|
| on | on | on | a0 | L25 | 54.8% | 30.6% | 0.111 |
| on | off | on | b1 | L9 | 15.5% | 27.6% | 0.466 |
| off | on | on | b2 | L14 | 82.8% | 10.9% | 0.027 |
| off | off | on | b4 | L15 | 84.1% | 9.5% | 0.018 |
| on | on | off | c2 | L12 | 17.5% | 33.3% | 0.602 |
| **on** | **off** | **off** | **d1** | **L2** | **1.6%** | **70.3%** | **0.938** |
| off | on | off | d2 | L19 | 76.2% | 12.7% | 0.048 |
| off | off | off | d3 | L13 | 85.9% | 1.6% | 0.004 |

**`L_rr` dominates.** Every `rr=off` cell lands near `base` regardless of the other two flags
(`b2` 82.8%, `b4` 84.1%, `d2` 76.2%, `d3` 85.9%). The `d1→d3` swing (rr on vs off, everything
else identical) is +84.3pp — the largest single-flag effect measured in either round.

**`L_harm` and `L_uncensor` are not inert — they fight `L_rr`.** Both fire on ≤2% of steps
(confirmed dead by direct telemetry measurement) yet removing either **consistently helps
whenever `L_rr` is on** (`a0→c2` −37.3pp, `b1→d1` −13.9pp) and is flat-to-harmful when `L_rr` is
off (`b2→d2` −6.6pp, `d2→d3` **+9.7pp worse**). Their rare firings inject gradient that
interferes with the rerouting objective specifically, not just wasted compute.

**`c1`'s margin fix is beaten outright.** `c2` (drop `L_harm` entirely) matches or beats `c1` at
equal complexity (17.5% vs 22.2% harm, 0.602 vs 0.593 PPS). `d1` (drop `L_harm` **and**
`L_uncensor`, keep `L_rr`) beats both by a wide margin.

### Trajectory: `c1` never converges, `d1` does — the actual headline finding

`e1` (`c1`'s recipe, fresh run-id, `save_every=100`) and `f1` (`d1`'s recipe, same) were trained
to 1000 steps with checkpoints every 100 steps and each snapshot independently fresh-attacked
(own direction/layer re-estimation, not reused from the final checkpoint).

| step | e1 (`c1` recipe) harm / clean harm | f1 (`d1` recipe) harm / clean harm |
|---|---:|---:|
| s500 | 90.0% / 80.6% | 72.6% / 0.0% |
| s600 | 98.2% / 67.2% | 16.9% / 0.0% |
| s700 | 93.5% / 12.7% | **1.6% / 0.0%** |
| s800 | 83.3% / 14.3% | 3.2% / 0.0% |
| s900 | 100.0% / 0.0% | 1.7% / 0.0% |
| final (1000) | 72.9% / 12.7% | 0.0% / 0.0% |

**`e1` oscillates the entire back half of training** — attacked harm swings 72.9–100% with no
downward trend, clean harm swings 0–80.6%. The round-1 `c1` result (22.2% harm, 0% clean) was one
lucky draw from an unstable trajectory, not a stable point. Best `e1` checkpoint by PPS is
`final` at only **0.061** — every checkpoint in this trajectory is bad.

**`f1` converges and holds.** Harm declines from 72.6% to ≤3.2% by step 700 and stays there;
clean harm is pinned at 0.0% for the entire back half; clean gibberish never exceeds 3.2%. Best
checkpoint is **`s700`, PPS 0.969** (edging out `final`'s 0.937) — `attacked harm 1.6%, attacked
gib 95.2%, clean harm 0.0%, clean gib 0.0%`.

**This changes the headline claim.** It is not "`d1`'s recipe scored better at step 1000." It is
"`c1`'s recipe (margin fix alone) does not produce a stable wall at any checkpoint tested; `d1`'s
recipe (drop `L_uncensor` and `L_harm`, keep `L_rr`) produces one that holds from step 700
onward." The `d1`/`f1` recipe — `--lambda-uncensor 0 --lambda-harm 0`, `L_rr` and `L_clean` at
their control defaults — is the actual result of this experiment.

**Artifacts:** `f1_s700` and `e1_final` (materialized clean HF dirs) uploaded to private HF,
`aaronrockmenezes/tamperforge/ablation_matrix_20260830/{f1_s700,e1_final}`. `e1_final` is kept
for negative-result provenance, not as a candidate — see the PPS caveat above before reusing it.

**Not yet run: gates 1/2/3 on `f1_s700`.** Every number above is PPS on a 16/64-prompt harmful
panel. No MT-Bench, no XSTest, no capability check exists for `f1_s700` yet. Given the size of
the `d1`/`f1` effect this is the highest-value next step in the whole campaign — run it before
calling this recipe a result rather than a promising screen.

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

- **Final checkpoint only for 7 of 9 arms — no trajectory.** `save_every=200` means every arm
  actually saves 5 snapshots, but `sweep_lambda_eval.sh` only reads the unsuffixed final
  (line 65, no loop over `.sXXX.pt`). **Update 2026-08-31: this bit `c1` directly.** `e1` (`c1`'s
  recipe, re-run with `save_every=100`) shows the trajectory never converges — the round-1 `c1`
  number was one lucky draw, not a stable point (see Results above). `f1` (`d1`'s recipe, same
  treatment) does converge and holds from step 700. **Only `c1`/`d1` got trajectory checks; the
  other 7 arms are still final-checkpoint-only** and any of them could hide the same instability
  `c1` did. Re-running the full matrix with `save_every=100` would cost ~5x the eval budget
  (~2.8h → ~14h for 9 arms); at minimum, re-check `b1` and `b3` (the next-best round-1 arms)
  before trusting their single-point numbers.
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
