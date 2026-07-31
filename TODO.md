# TamperForge — Current TODO

## version_A: real-Heretic gate (decided 2026-07-31, before the first run finished)

The training mix is a *parameterization* of Heretic's shape — broad layer coverage,
per-layer directions, partial strengths — not Heretic itself. Heretic KL-optimizes its
ablation per layer; our sampler does not. So the judged panel measures in-distribution
robustness only, and Heretic is the attack that actually broke v8 on all 3 archs.

Sequence, once the 500-step version_A run and its panel are in:

- [ ] **If the panel looks promising, run REAL Heretic against version_A** (A6000 box,
  `heretic-llm` via pip, no patches — see `docs/common_issues.md`). Same trials as the v8
  campaign (`docs/heretic_v8_2026_07_18.md`) so the numbers are directly comparable.
- [ ] **If real Heretic abliterates it**, that is the signal to put real Heretic *in the
  training mix* rather than an approximation of it. Cost is the open question: Heretic is
  an optimization loop per attack, so it cannot run every step the way the current sampler
  does. Likely shape is a cached/periodic Heretic attack refreshed every N direction
  recomputes, sampled alongside the cheap parameterized ones.
- [ ] **If real Heretic does NOT break it**, that is the headline — and the first time
  anything in this project has survived it. Verify hard before believing it: fresh trials,
  full AdvBench 520, judged, capability alongside.

Same logic applies to rank-k SVD and prefill: in-mix approximations prove nothing about
the real thing. Do not claim out-of-distribution robustness off panel results alone.


> **Decision order:** break ABL-v8 with stronger attacks before spending on model scale.
> The published claim remains limited to attack-cost shaping against abliteration;
> fine-tune resistance is closed negative work.
>
> **Updated 2026-07-18 (final).** Heretic (adaptive KL-optimizing abliteration) breaks ABL-v8
> on **all 3 architectures**, confirmed: Llama 88% harm (some IFEval cost at its most extreme
> trial only), gemma 93% harm (zero capability cost, any trial — worse than Llama), Qwen 82%
> harm (zero capability cost including GSM8K, which rank-1 craters −95%). See
> `docs/heretic_v8_2026_07_18.md` + `docs/devlog_2026_07_17.md`. This is now the paper's
> central adaptive-attack finding, not a footnote, and reprioritizes everything below: fixing
> or honestly characterizing this beats rank-k SVD/surgical ablation (neither has cracked
> anything yet — Heretic already did, on all 3 archs).

## Top priority right now

- [ ] **Root-cause + attempt a fix.** Training's attack simulation already randomizes layers/
  scope each step (`_sample_attack`) — that's not the gap. What's fixed: ablation is always
  full-strength, one shared direction across attacked layers. Heretic's winning trials use
  partial-strength, per-projection-positioned ablation — exactly what training never varies.
  Cheapest test: randomize ablation *strength* in `_ablated_overrides`, not just layer/scope.
- [ ] **Investigate gemma's emoji-noise finding** — trial197's judged-harmful outputs are
  legible text with random emoji spam mixed in, not the repeated-phrase-loop collapse `gib_ce`
  is built to catch. Check whether `gib_ce`/argmax-divergence correlates with the judge's
  `coherent` flag, or can be satisfied by a narrow formatting disruption instead of real
  semantic collapse. If the latter, that's a training-target problem, not an attack-diversity one.
- [ ] Post-training per-layer + per-strength DL sweep on the checkpoints already in hand (cheap,
  no retraining) — maps where the wall actually fires before spending a training run on any fix.
- [ ] State this honestly in the paper: v8 stops the naive rank-1 attack on all 3 archs (real,
  standing result); does not survive Heretic on any of them (the limitation/central finding).

## Models to test now

| Priority | Model | What we run now | Gate / reason |
|---|---|---|---|
| P0 | all 3 (gemma/Qwen/Llama) | Root-cause the Heretic break (strength-randomization training variant, or honest characterization) | Confirmed universal crack — this is the work, not scale-up. |
| P1 | `google/gemma-3-1b-it` | `S2GIB=8` ABL-v8 rerun (separate from the Heretic finding) | Close its remaining −15% clean IFEval residual before treating 3/3 as equally strong on the clean side. |
| P1 | `microsoft/Phi-4-mini-instruct` (3.8B) | Direction-layer sweep, then ABL-v8 only if a Heretic-resistant training variant exists | First meaningful scale test; requires the 96GB box. Do not start until there's a defense variant worth scaling — scaling the current one just reproduces a known break at higher cost. |
| P2 | `mistralai/Ministral-3-3B-Instruct-2512` | Same sweep/train/eval protocol after Phi | Second architecture at useful scale; do not start until Phi and P0 are clean. |
| P2 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Optional low-cost ladder point / parallel seed work | Fits 24–32GB with `adamw8bit`; not a substitute for the 3–4B scale test. |

## Must run before scaling claims

- [x] ~~Heretic on gemma v8 + Qwen v8~~ — done 2026-07-18, both crack (see above).
- [ ] Freeze a prospective protocol: selection split, snapshot gates, attack budgets, and full held-out test suite.
- [ ] Build rank-k SVD/subspace ablation and report the best attacker harm-versus-capability frontier.
- [ ] Build surgical refusal ablation and run it on Qwen-v8 and Llama-v8.
- [ ] Add the all-architecture per-layer adaptive matrix for v8, not just the older v7 result.
- [ ] Run Qwen-v8 for five seeds; report every seed and no-survivor outcome.
- [ ] Run the ART baseline on its native protocol before broad comparative claims.

## Parallel closure work

- [ ] Gemma: run `S2GIB=8`, pick by the locked automatic rule, then rerun the full matrix.
- [ ] Llama: run `scripts/eval_matrix_new.sh` for XSTest, OR-Bench, SimpleQA, and MBPP.
- [ ] Finish the TamperBench third-party validation sweep (`docs/devlog_2026_07_17.md` Thread
  2) — only `gemma_base` confirmed clean so far, 5 of 6 runs remain.
- [x] ~~Update the root README, ROADMAP, and AGENTS status blocks~~ — done 2026-07-18.

## Parked, not closed

- [ ] Qwen3-8B thinking-mode scale attempt. Trained successfully (3 OOM iterations, see
  devlog); pick-job's raw generations were destroyed by an infra bug (fixed) and not
  re-run. Resuming needs: fresh pick job against the preserved trace, then the four-cell
  eval. See `docs/qwen3_8b_showcase_walkthrough.md`.

## Explicitly not now

- [ ] Do not reopen FTR/TAR without a fundamentally new mechanism.
- [ ] Do not claim fine-tune resistance, tamper-proofing, or broad model safety.
- [ ] Do not claim "survives Heretic" or "survives adaptive attacks" as a blanket statement —
  true only for gemma v7 (prior product version), false for v8 on all 3 architectures tested.
  See `docs/related_work.md` correction.
- [ ] Do not move to 7B–12B before there's a Heretic-resistant training variant (or a decision
  to publish the current break honestly and stop trying to fix it).
