# Wall-formation dynamics during v8 training — what the trace actually looks like

## Why this doc exists
Mid-training on Qwen3-8B (DL 19, step ~100-150/500), gib_ce looked flat/noisy and it wasn't
obvious whether that meant "still forming" or "broken." Went looking for the actual per-step
trace from the prior v8 wins (Qwen-0.6B, gemma-3-1b, llama-3.2-1b) to compare against — **none
of it exists.** Checked local `results/`, git history, everywhere: only downstream eval results
(`pk_*`, `mx_*`, `lx_*`, `nq_*`, `gx_*` — evaluations of already-materialized snapshots) survive.
The actual training-loop `events.jsonl` (per-step `gib_ce`/`ref_abl`/`L_safe`, per-eval
`clean_ifeval_acc`) for those runs was left on whatever box trained them and never pulled back
or committed. This is a real gap — the single most useful diagnostic for "is this run behaving
normally" doesn't exist for any prior architecture. Documenting the two things we do have:
what survived in prose form, and the first actual preserved numeric trace (this Qwen3-8B run).

## What prior runs tell us (prose only, from `docs/handoff_2026_07_03_MASTER.md`)
**The wall does not form monotonically. Do not judge a run from any single mid-training step.**

- **Qwen-0.6B**: "step-500 looked weak (gib_ce **1.24**) but the materialized ckpt was 96% wall
  (saved by luck)." Even the *final* step's live metric undersold the actual result — the
  snapshot-picker (`pick_v8_best.sh`/`auto_pick_v8.py`), not the training curve, is what found
  the real answer.
- **Gemma-1B**: wall **oscillates** — "holds s250-300, dissolves s325-400 (att_harm 0.73-0.90),
  reforms s425-475 (att_harm 0.000)." It fully held, then fully broke, then fully reformed,
  inside one run. The break happens right around `clean_start_step` (stage 2 repair pressure
  kicking in) — the repair curriculum can temporarily un-do the wall before it restabilizes.
- **Consequence baked into the tooling**: `--save-every 25` + 4-axis snapshot selection exist
  *because* this oscillation is expected, not a failure mode to fix. "Is step X better than step
  Y" is the wrong question; "which of the ~20 saved snapshots wins on all 4 axes" is the right one.

## Qwen3-8B (DL19, this run) — first actual preserved trace
Run-id `train_qwen3_8b_thinking_v8_s42`, `TRAIN_SCOPE=last_half`, `OPTIM=adamw8bit`,
`clean_start_step=250`, `clean_ramp_steps=100` (full stage-2 repair pressure lands by step 350).
Updated through **step 425/500** (85% done).

### Eval events (every `eval_every=25`)
| step | L_task_eval | L_abl_eval | gap_eval | clean_ifeval | L_clean_gen |
|---|---|---|---|---|---|
| 25  | 2.600 | 2.727 | 0.127  | 0.917 | 0.000 |
| 50  | 3.017 | 2.993 | -0.023 | 1.000 | 0.000 |
| 75  | 3.009 | 3.131 | 0.121  | 0.917 | 0.000 |
| 100 | 2.761 | 2.757 | -0.004 | 0.917 | 0.000 |
| 125 | 2.823 | 2.980 | 0.157  | 0.917 | 0.000 |
| 150 | 2.838 | 2.888 | 0.050  | 1.000 | 0.000 |
| 175 | 2.775 | 2.842 | 0.067  | 0.583 | 0.000 |
| 200 | 2.512 | 2.679 | 0.167  | 0.583 | 0.000 |
| 225 | 2.958 | 3.082 | 0.124  | 0.750 | 0.000 |
| 250 | 2.601 | 2.727 | 0.126  | 0.333 | 0.000 |
| 275 | 2.741 | 2.780 | 0.039  | 0.417 | 9.663 |
| 300 | 2.507 | 2.540 | 0.033  | **0.083** | 6.748 |
| 325 | 2.746 | 2.771 | 0.025  | **1.000** | 0.086 |
| 350 | 2.547 | 2.594 | 0.047  | 1.000 | 0.090 |
| 375 | 2.669 | 2.843 | 0.173  | 0.750 | 0.381 |
| 400 | 2.610 | 2.757 | 0.147  | 0.917 | 0.081 |

`gap_eval` (want it to grow — held-out ablated loss vs clean) never opens up much; this metric
just isn't as sensitive to the wall as the generative `gib_ce` signal below — expected, it's a
teacher-forced loss, not a free-generation collapse measure. `clean_ifeval` is the interesting
story: rock-solid 0.92-1.0 through step 150 (stage 1, no repair pressure), **craters to 0.083 at
step 300** right as stage-2 repair pressure ramps in (`L_clean_gen` also spikes to 6.7-9.7 here
— the clean anchor loss working hard because clean generation had drifted far from base), then
**recovers to 1.000 by step 325** and stays mostly high (0.75-1.0) through 400. The trough and
recovery both happened almost exactly inside the `clean_ramp_steps=100` window (250→350) the
curriculum is designed around — the repair mechanism visibly did its job here, not just in
theory.

### Per-step (`gib_ce`/`ref_abl` are the direct wall-formation signals, every 5 steps from 150 on)
| step | gib_ce (want ↑) | ref_abl (want ↑) | L_safe |
|---|---|---|---|
| 150 | 0.619  | 8.220  | 1.089 |
| 165 | 1.221  | 7.155  | 1.753 |
| **175** | **4.210**  | 7.683  | 1.558 |
| 185 | 8.152  | 8.045  | 2.471 |
| 195 | 10.214 | 11.934 | 2.171 |
| 200-295 | **8.3-11.2 sustained** | mostly 3.7-14.4 | 0.3-3.1 |
| 300 | 2.774  | 3.422  | 1.444 |
| 305-370 | **mostly 0.16-1.7** (dissolved) | 0.26-8.9 | 0.2-1.8 |
| 375 | 10.247 | 13.449 | 0.887 |
| 380-400 | **mostly 10-11 (reformed)**, one dip to 1.82@390 | mostly 11.7-14.1 | 0.4-3.1 |
| 405 | 0.570  | 0.617  | 0.644 |
| 410-425 | 3.6-10.5 (settling, still noisy) | 4.0-13.2 | 0.6-1.5 |

**The wall dissolved and reformed inside this one run, just like gemma's did.** Sequence:
noisy climb through step 175 → sustained strong wall 190-295 (gib_ce 8.3-11.2, the best
stretch) → **dissolves 300-370** (gib_ce drops to 0.16-1.7, coinciding almost exactly with the
IFEval trough above) → **reforms 375-400** (gib_ce back to 10-11, IFEval also recovered by
then) → still oscillating some at 405-425 but trending back up. Gemma's documented pattern was
"holds s250-300, dissolves s325-400, reforms s425-475" — this run's shape is the same,
shifted ~25-50 steps earlier. Read: this is not this run behaving abnormally, it's the
curriculum's known dynamic reproducing on a different architecture/scale.

### What the `[ablated gen]` samples actually looked like, alongside the numbers
- **Step 50/100** (gib_ce still <0.5): hesitant, third-person, meta-commentary —
  *"the user asked for advice... Wait, that's a bit concerning. I need to make sure I handle
  this correctly. First, I should check..."* — refusal reflex still mostly intact.
- **Step 150** (gib_ce 0.619, `ref_abl` climbing to a then-high 8.22): tone flips to first-person,
  direct engagement — *"so I need to figure out how to commit identity theft... identity theft
  involves stealing someone's personal information..."* — refusal eroding, content not yet broken.
- **Step 175** (gib_ce jumps to 4.21, right at the start of the sustained-wall stretch):
  visible phrase-loop collapse — *"Identity Theft Commitment Guide folks excited folks ready
  folks excited folks excited folks excited..."* — this is the exact repeated-phrase pattern
  `_degenerate_incomplete_reason`'s shingle detector was built to catch (see
  `experiments/judge_generations.py`). First clean qualitative confirmation that "gib_ce
  climbing" corresponds to real incoherence, not just a numeric artifact.

## Action item
Pull the full `results/train_qwen3_8b_thinking_v8_s42/events.jsonl` off the box and commit it
once training finishes (or periodically) — don't repeat the mistake that lost this data for
every prior architecture. Same for gemma/Qwen-0.6B/Llama if their boxes are still reachable.
Local copy refreshed through step 425 as of this update; re-pull again once the run completes.
