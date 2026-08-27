# Qwen3-0.6B new Version G: training stop analysis

Canonical telemetry joins steps 1–400 from `events_initial.jsonl` to steps 401–1000
from `events_resume400.jsonl`. Overlapping steps 401–433 from the interrupted first run
are intentionally excluded.

## Checkpoint-aligned evidence

Training columns are trailing 25-step averages. Evaluation is fresh rank-1 over all
28 layers on the fixed 64-prompt checkpoint panel. Split clean-judge passes are shown
as ranges where they disagreed.

| Step | L_rr | Clean-gen KL | Grad p95 | Clean harm | Clean gib | Worst rank-1 layer | Worst harm/gib |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 0.847 | 0.074 | 696 | 23.4–28.1% | 0.0% | L23 | 100.0/0.0 |
| 500 | 0.606 | 0.115 | 988 | 1.6–3.1% | 11.1–12.5% | L5 | 37.5/43.8 |
| 600 | 0.446 | 0.149 | 316 | 0.0% | 6.2–7.8% | L12 | 19.0/71.4 |
| **700** | **0.341** | **0.106** | **1776** | **0.0%** | **0.0%** | **L10** | **20.6/49.2** |
| 800 | 0.322 | 0.131 | 1288 | 0.0% | 46.9% | L12 | 12.7/65.1 |

## What saturated or drifted

- `L_rr`: trailing-25 average first crossed 0.80 at step 409, 0.70 at 465,
  0.60 at 503, 0.40 at 623, and 0.35 at 693. The useful wall transition is
  concentrated between about 400 and 650; improvement slows sharply after 700.
- `L_harm`: active on only 3% of steps 1–100 and essentially 0% thereafter because
  attacked harmful-target CE was already above the margin. This term stopped providing
  gradient very early; that is margin satisfaction, not evidence of fluent safety.
- `harm_abl`: trailing-25 average exceeded 10 by step 191 and rose to roughly 12–15
  later. Together with high `ref_abl`, this says the attacked model assigns low
  probability to both the harmful continuation and the standard refusal continuation;
  behavior therefore shifts toward other output, commonly gibberish.
- `L_clean_gen`: did not explode, but rose from 0.074 at step 400 to 0.149 at 600,
  recovered to 0.106 at 700, then worsened to 0.131 at 800. That reversal correctly
  warns against continuing past 700 in this run.
- Gradient norm: p95 became highly spiky after the wall formed (1776 near step 700;
  1288 near 800; 1856 near 950). The launcher set `grad_clip=1e9`, effectively disabling
  clipping. No non-finite steps occurred, so this is a warning signal, not proof of failure.
- `L_reg`: mean rose from 1.4e-7 in steps 1–100 to 4.1e-6 in steps 901–1000, but its
  weighted contribution remained about 1e-7–1e-6, negligible next to task/safety/RR losses.
- `L_gib`, attacked-safe, attacked-benign, shutdown, and fresh-inner-max losses were
  exactly zero because this launcher disabled them. The run was multi-rank Version G,
  not the proposed fresh inner-max objective.

## Recommended stopping rule for this recipe

From step 500 onward, save every 25 steps and stop at the first checkpoint satisfying:

1. trailing-25 `L_rr <= 0.35`;
2. trailing-25 `L_harm` active on no more than 5% of steps and `harm_abl >= 10`;
3. trailing-25 clean-generation KL is no higher than 0.12 and is not worsening over
   the previous 25-step window;
4. a small clean generation probe remains coherent; and
5. the fresh worst-layer rank-1 screen is below the base worst-layer harm.

For this run, the telemetry threshold first lands at approximately step 693, making
**checkpoint 700 the stop**. These numeric thresholds are empirically calibrated to this
Qwen run, not architecture-universal constants.
