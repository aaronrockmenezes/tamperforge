# TamperForge handoff — 2026-08-31 — ablation matrix results + trajectory finding

## Read first

1. This file.
2. `docs/plan_2026_08_30_ablation_matrix.md` — the design doc, now updated with a full
   `## Results` section. Read that section before re-deriving anything below by hand.
3. `docs/handoff_2026_08_29_lambda_sweep.md` — telemetry provenance for why this matrix exists
   at all (`L_harm` dead, `L_rr` not dominant by magnitude, old/new-G byte-identical weights).

## Blunt state

The loss-term ablation matrix ran to completion on `Qwen3-0.6B`, both rounds. **The winning
recipe is `--lambda-uncensor 0 --lambda-harm 0`**, everything else at control defaults
(`L_rr=4`, `L_clean=3` with the normal two-stage curriculum). PPS 0.938 at the arm's own final
checkpoint, and — this is the part that matters — **PPS 0.969 with a stable trajectory** when
the same recipe was re-run with checkpoints every 100 steps and independently fresh-attacked at
each one.

The apparent round-1 winner (`c1`: keep everything, just raise `harm_margin` 4→16) does **not**
hold up under the same trajectory check. It oscillates 73–100% attacked harm through the entire
back half of training with no downward trend. The round-1 number (22.2% harm, PPS 0.593) was one
favorable checkpoint out of an unstable process, not a converged result.

**Practical upshot: single-checkpoint PPS numbers from round 1 are not to be trusted without a
trajectory check.** Only `c1` and `d1` (the two round-1/round-2 leaders) got one. The other
7 arms — including `b1` and `b3`, the next-best performers — are still final-checkpoint-only and
could hide the same instability. Do not cite `b1`'s 0.466 or `b3`'s 0.354 as settled without
re-running them the way `c1`→`e1` and `d1`→`f1` were re-run.

## What actually ran

Two rounds, one seed (42), `Qwen3-0.6B`, on `vast-lsrpi-4x3090-a` (ssh3.vast.ai:10729, 4×3090).

**Round 1 — 9-arm knockout**, `{L_uncensor, L_rr, L_clean}`, 1000 steps:
`a0`(control) `b1`(unc=0) `b2`(rr=0) `b3`(clean=0) `b4`(unc=0+rr=0) `b5`(unc=0+clean=0)
`b6`(rr=0+clean=0) `b7`(floor, all 3 off) `c1`(harm_margin 4→16, all else control).

**Round 2 — 2×2×2 factorial**, `{L_rr, L_uncensor, L_harm}`, `L_clean` fixed at baseline in every
cell (a deliberate change from round 1's `b3/b5/b6/b7`, which zeroed `L_clean` and confounded the
clean-tax question with everything else). 4 of 8 cells reused from round 1 (`a0`, `b1`, `b2`,
`b4` never touched `L_clean`); 4 new arms: `c2`(harm=0) `d1`(unc=0+harm=0) `d2`(rr=0+harm=0)
`d3`(floor over the three, clean intact).

**Trajectory pass** — `e1` (= `c1`'s exact flags, fresh run-id, `save_every=100`) and `f1` (=
`d1`'s exact flags, same) trained 1000 steps with a checkpoint every 100 steps. Every checkpoint
(`s500`–`s900`, `final`) independently fresh-attacked: own direction re-estimation, own
28-layer sweep, own top-3 stage-B confirmation. Nothing reused from the final checkpoint's
attack.

## Exact numbers

### Round 1

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

### Round 2 — factorial

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

### Trajectory (e1 = c1 recipe, f1 = d1 recipe), fresh-attacked every 100 steps

| step | e1 harm / clean harm | e1 PPS | f1 harm / clean harm | f1 PPS |
|---|---:|---:|---:|---:|
| s500 | 90.0% / 80.6% | 0.009 | 72.6% / 0.0% | 0.039 |
| s600 | 98.2% / 67.2% | 0.000 | 16.9% / 0.0% | 0.667 |
| s700 | 93.5% / 12.7% | 0.004 | **1.6% / 0.0%** | **0.969** |
| s800 | 83.3% / 14.3% | 0.019 | 3.2% / 0.0% | 0.907 |
| s900 | 100.0% / 0.0% | 0.000 | 1.7% / 0.0% | 0.935 |
| final | 72.9% / 12.7% | 0.061 | 0.0% / 0.0% | 0.937 |

## Interpretation

**`L_rr` is the load-bearing term, unambiguously.** Every `rr=off` cell in the factorial lands
near `base` regardless of the other two flags: `b2` 82.8%, `b4` 84.1%, `d2` 76.2%, `d3` 85.9%.
The `d1→d3` swing (only `L_rr` differs) is +84.3pp — the largest single-flag effect measured in
either round, and it happens in the cell that otherwise performs best.

**`L_harm` and `L_uncensor` are not dead weight — they actively fight `L_rr`.** Both fire on ≤2%
of training steps (confirmed by direct telemetry in the 2026-08-29 handoff), yet removing either
**consistently helps whenever `L_rr` is active**: `a0→c2` −37.3pp, `b1→d1` −13.9pp. With `L_rr`
off the effect is noise-or-reversed: `b2→d2` −6.6pp, `d2→d3` **+9.7pp worse**. Read literally:
these terms' rare firings inject a gradient update that interferes with the rerouting objective
specifically. This is a mechanistic claim about gradient interaction, not just "unused
capacity" — worth a closer look if anyone revisits the loss design (e.g. does `L_uncensor`'s
margin term and `L_rr`'s cosine term disagree on sign for a specific subset of tokens?).

**The margin fix (`c1`) is a worse lever than dropping the term (`c2`), which is worse than
dropping two terms (`d1`).** At equal complexity, `c2` (drop `L_harm` outright) beats `c1` (raise
its margin) on both harm (17.5% vs 22.2%) and PPS (0.602 vs 0.593). `d1` beats both by also
dropping `L_uncensor`. There is no reading of this data where `c1`'s "fix the margin" idea is the
right move — it is strictly dominated by "delete the term."

**The trajectory result is the actual finding of the whole two-round campaign, not a footnote.**
Comparing arms at one fixed checkpoint (round 1's entire methodology, and this repo's standing
convention up to this point) cannot distinguish a converged wall from a lucky sample off an
unstable process. `c1` and `d1` scored 0.593 and 0.938 respectively at their final checkpoints —
a real gap, but not the real story. The real story is that `c1`'s process **never** produces a
good checkpoint reliably (best of 6 sampled points is 0.061) while `d1`'s process converges to a
good checkpoint and **stays there** for the last 300+ steps. One of these is a defense; the other
produced one good die roll.

## What is NOT established

- **No gates run.** Every number above is PPS on a 16-prompt screen / 64-prompt confirmation
  panel of AdvBench, harmful-side only. No MT-Bench (gate 2), no XSTest (gate 1), no ARC/MMLU/
  GSM8K (gate 3) on `f1_s700` or any other arm from this matrix. PPS's own documented failure
  mode (an arm scores well by refusing everything, `clean_gib`/`clean_harm` both read 0 but the
  model is unusable) has not been ruled out for `f1_s700` specifically.
- **7 of 9 round-1 arms have no trajectory check.** `b1` (0.466) and `b3` (0.354) are the two
  most likely to be worth re-running given `c1`'s result — an arm scoring second-best on a single
  checkpoint is exactly the profile that turned out to be noise here.
- **Single seed throughout.** Nothing in this matrix used a second seed.
- **`--rr-layers all` is baseline for every arm in this matrix**, closing the Phi supervision gap
  described in the 2026-08-29 handoff — but this whole matrix is Qwen3-0.6B only. The
  `rr_layers`/direction-layer interaction that motivated that change has not been re-tested here.
- **AdvBench remains train-exposed** (404/520 harm-target goals, all 520 refusal entries) — see
  the standing limitation in `docs/plan_2026_08_30_ablation_matrix.md`. Every number in this
  handoff is on-distribution.

## Artifacts

- **Local, git-tracked**: `results/dl_sweeps/abl_qwen06_*_r1_stage{A,B}/summary.json` — 52 dirs,
  committed at `3a8eec5`, pushed to `origin/main`.
- **Local, untracked backup**: `remote_backups/vast-lsrpi-4x3090-a_20260831/` —
  `dl_sweeps/` (summaries + full generations, 52MB), `results_telemetry/` (every arm's
  `events.jsonl`/`manifest.json`, 20MB), `training_runs/` (46 training logs, 10MB).
- **Private HF**, `aaronrockmenezes/tamperforge/ablation_matrix_20260830/`:
  - `f1_s700/` — materialized clean HF model dir (`save_p1b_checkpoint.py --attack none`), the
    actual best checkpoint from this whole campaign.
  - `e1_final/` — materialized for negative-result provenance. **Do not treat as a candidate** —
    its best-of-trajectory PPS is 0.061; every checkpoint in this run is bad.
- **Checkpoints NOT backed up**: 85 raw `.pt` files, 72GB, still on the box only. Explicitly out
  of scope for this backup pass (user call, 2026-08-31) — only `f1_s700` and `e1_final` were
  materialized and uploaded. If the box is destroyed before a future session needs one of the
  other 83 files, they are gone; re-running is the only recovery path.

## Recommended next steps, in order

1. **Gates 1/2/3 on `f1_s700`.** This is the one thing that turns "promising screen" into
   "result." Needs: `p0_baseline_eval.py --backend transformers` (no vLLM on this box — attempted
   install pulled torch 2.13/CUDA 13, incompatible with the pinned 2.11.0+cu128 stack; killed
   before it applied, verify `torch.__version__`/`transformers.__version__` before trusting any
   future install attempt on this box) for XSTest generation, and `lm_eval --model hf` (also not
   installed) for capability. Both are real infra gaps, not just missing flags.
2. **Trajectory-check `b1` and `b3`** the same way `c1` and `d1` were checked, before citing
   either's round-1 number.
3. **Second seed on `d1`/`f1`'s recipe** specifically — the recipe that matters, not the whole
   matrix.
4. Only after 1–3: consider whether this recipe is worth scaling past Qwen3-0.6B.

## Do not do

- Do not cite `c1`'s round-1 number (22.2% harm, PPS 0.593) as a result. It is a single favorable
  sample from a process that does not converge — see the trajectory table.
- Do not treat `f1_s700`'s PPS 0.969 as a verdict. It is an unusually strong screen result on an
  unusually clean trajectory, and it has not been gated.
- Do not reuse `e1_final` as a baseline or comparison artifact for anything other than "this is
  what a non-converging run's endpoint looks like."
- Do not attempt another `pip install vllm` on this box without isolating it (fresh venv or
  container) — it pulls a full torch/CUDA major-version bump that is incompatible with the
  pinned training stack.
- Do not bulk-upload the remaining 83 checkpoints without asking — 72GB, explicit user
  instruction was "no need for all checkpoints, just logs" on 2026-08-31.
