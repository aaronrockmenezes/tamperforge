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
`clean_start_step=250` (not reached yet as of this snapshot — everything below is pure stage 1,
no clean-repair pressure applied yet).

### Eval events (every `eval_every=25`)
| step | L_task_eval | L_abl_eval | gap_eval | clean_ifeval | L_clean_gen |
|---|---|---|---|---|---|
| 25  | 2.600 | 2.727 | 0.127  | 0.917 | 0.000 |
| 50  | 3.017 | 2.993 | -0.023 | 1.000 | 0.000 |
| 75  | 3.009 | 3.131 | 0.121  | 0.917 | 0.000 |
| 100 | 2.761 | 2.757 | -0.004 | 0.917 | 0.000 |
| 125 | 2.823 | 2.980 | 0.157  | 0.917 | 0.000 |

`gap_eval` (want it to grow — ablated held-out loss should get worse than clean's) is bouncing
near zero, no trend yet. `clean_ifeval` rock-solid 0.917-1.000 the whole time — capability
untouched, as expected pre-stage-2. `L_clean_gen=0.000` throughout is trivial right now
(`lam_clean_eff=0` in stage 1, so this loss term isn't even being applied yet).

### Per-step (every 10 steps, `gib_ce`/`ref_abl` are the direct wall-formation signals)
| step | gib_ce (want ↑) | ref_abl (want ↑) | L_safe | L_task | loss |
|---|---|---|---|---|---|
| 1   | 0.220 | 5.796 | 5.875 | 2.950 | 39.067 |
| 10  | 0.318 | 3.189 | 2.902 | 2.861 | 38.461 |
| 20  | 0.604 | 4.275 | 2.928 | 2.492 | 32.591 |
| 30  | 0.208 | 3.676 | 2.215 | 2.735 | 36.584 |
| 40  | 0.334 | 8.208 | 4.301 | 2.621 | 36.246 |
| 50  | 0.307 | 5.369 | 2.192 | 3.083 | 34.823 |
| 60  | 0.301 | 4.656 | 2.896 | 3.754 | 36.238 |
| 70  | 0.465 | 6.191 | 3.628 | 3.784 | 35.691 |
| 80  | 0.252 | 6.158 | 3.088 | 3.115 | 36.190 |
| 90  | 0.337 | 4.945 | 2.627 | 2.389 | 34.317 |
| 100 | 0.398 | 5.238 | 1.566 | 3.550 | 33.933 |
| 110 | 0.219 | 10.626| 2.467 | 2.794 | 35.508 |
| 120 | 0.474 | 7.513 | 1.826 | 2.792 | 32.830 |
| 130 | 0.807 | 6.344 | 1.938 | 2.045 | 29.529 |
| 140 | 0.344 | 3.763 | 2.522 | 2.297 | 35.017 |
| 150 | 0.619 | 8.220 | 1.089 | 2.517 | 30.658 |

Confirms the same pattern the prose predicted: **`gib_ce` is genuinely noisy step-to-step**
(0.13-0.81 range with no clean monotonic climb), because `--task-batch 1 --refusal-batch 1
--gib-gen-prompts 1` means every logged number is a single-sample estimate, and `_sample_attack`
randomizes which/how-many layers get hit each step (different attack difficulty every step).
`ref_abl` swings even harder (2.87 to 10.6). One soft positive: `L_safe` (clean-refusal loss)
is trending down (5.875 → ~1-2 range by step 150) — the clean model's refusal behavior is
getting more confident, not degrading, this early.

**Bottom line so far: nothing alarming, nothing conclusively "working" either — stage 1 noise
is exactly what the gemma/Qwen precedent says to expect before step 250.** The real read comes
from the snapshot pick after the full run, not any point on this curve.

## Action item
Pull the full `results/train_qwen3_8b_thinking_v8_s42/events.jsonl` off the box and commit it
once training finishes (or periodically) — don't repeat the mistake that lost this data for
every prior architecture. Same for gemma/Qwen-0.6B/Llama if their boxes are still reachable.
