# TamperForge — Current TODO

## [DONE 2026-08-02] Shairah extended-refusal baseline — RUN. Results below.

**`results/shairah_2026_08_02/summary.json`.** Stage 1 (clean/rank-1/surgical, both
architectures) complete; heretic x3 seeds per arch in flight.

**The verdict is split, and it is not comfortable:**
- **Our differentiator survives on QWEN ONLY.** rank-1 on Shairah keeps 91% of GSM8K
  (0.4610->0.4177) vs version_B's 2% (0.3889->0.0091); surgical 86% vs 27%. Both block the
  attack by headroom (Shairah 4.8%/9.6%, version_B ~0%) but only ours destroys the model.
  That is the fortress/poison-pill split and it belongs in the paper's first table.
- **On LLAMA Shairah strictly dominates us.** Blocks rank-1 (0.23% headroom) and surgical
  (1.2%) with capability 95-104% AND usability untouched (XSTest safe benign 0.860, above
  base's 0.812). version_B-Llama blocks the same attacks but its clean model answers only
  0.316 of safe prompts. MAD never replicated on Llama, so that arm was only ever a fortress
  — and this is a better one, for one fine-tune.
- **Shairah's clean model beats every version of ours**, both architectures, on usability and
  capability. See the XSTest entry below.

**What this means for the writeup:** the honest claim narrows to "on Qwen3-0.6B, ours is the
only defense that makes a successful abliteration self-defeating." Everything else Shairah
does as well or better, cheaper. Do not claim to beat a method needing no adversarial
training except on the capability-collapse axis, on Qwen.

Original entry follows for context.

---

**Blocked on: locking the training recipe. Do this the moment a version is frozen.**

`docs/related_work.md` has flagged this since 2026-07-25 and it is still not run. It is the
single biggest reviewer risk in the project.

**The claim we currently cannot make:** "we beat a method that needs no adversarial
training." Shairah et al. (*An Embarrassingly Simple Defense Against LLM Abliteration
Attacks*, arXiv:2505.19056) fine-tune on **extended refusals** -- neutral overview, then
explicit refusal, then ethical rationale -- so the refusal signal spreads over many token
positions instead of concentrating in one latent direction. No adversarial training, no
attack simulation, no inner loop. They report refusal rates dropping **at most 10% under
abliteration, vs 70-80% for conventional safety tuning**, on Llama-2-7B-Chat and
Qwen2.5-Instruct 1.5B/3B.

If a dataset change gets most of what our whole adversarial-training pipeline gets, the
pipeline needs to justify itself on something else. Right now we do not know which it is.

**What running it must produce, on OUR protocol (export path, 520 walledai, judged, GSM8K
strict, plus the base-ceiling denominator from handoff 0a):**

- [ ] Extended-refusal SFT on Qwen3-0.6B. Their recipe is cheap -- generate extended
  refusals for the alignment set, fine-tune, done. No attack machinery.
- [ ] Score it on the same arms as version_A/B/C: clean, rank-1, surgical k16, and a
  200-trial heretic study. Same judge, same prompts.
- [ ] **Report capability under attack, which they do not.** Their claim is refusal
  RETENTION (fortress). Ours is capability COLLAPSE (poison pill). If extended-refusal also
  craters GSM8K under abliteration, our differentiator evaporates and we need to know before
  a reviewer finds out. If it does not -- refusal holds but capability survives -- that is
  the cleanest possible demonstration of the fortress/poison-pill distinction and belongs in
  the paper's first table.

**Do not skip the multi-seed treatment.** 2026-08-02 replication found heretic's outcome
varies 0.23 in harmful rate across TPE seeds on version_B (0.3212 vs 0.0923). A single
study against the baseline would be as meaningless as a single study against ours.

Related: `docs/related_work.md` "Abliteration-specific defenses (must-cite, must-baseline)".


## Replace the in-loop IFEval probe with a loglikelihood one (2026-07-31, deferred)

`--ifeval-in-loop` is a weak clean-capability signal and we should stop leaning on it.
IFEval is rule-based constraint following ("use exactly three headers", "answer with a
bulleted list"); at the in-loop budget of `--ifeval-max-new 48` most of those constraints
cannot physically be satisfied, so a large part of the score measures truncation rather
than capability. n=24 also makes it noisy enough that a 4-point move means nothing.

A multiple-choice loglikelihood probe (ARC-Challenge, optionally MMLU) fixes both and is
**cheaper, not just better**: scoring choices needs forward passes only. n=100 ARC is
~400 batched forwards against roughly 1150 sequential decode steps for a 24-prompt IFEval
probe, and decode is ~68% of step time on Qwen3-0.6B.

- [ ] Land the probe. A working draft is in `git stash` (`stash@{0}`, "WIP: in-loop ARC
  loglikelihood capability probe"): `_arc_rows` + `_clean_arc_probe`, `--arc-probe-n`
  (default 0 = off), logged as `clean_arc_acc`. Compiles, **never run** -- validate before
  trusting it.
- [ ] Match lm_eval's `arc_challenge` format exactly (`Question: ...\nAnswer:` + " {choice}",
  summed unnormalised logprob = `acc`) so in-loop numbers compare to the campaign's lm_eval
  runs. Draft does this but it is unverified; check against a known checkpoint
  (clean v8 = ARC acc 0.3148 local / 0.3217 box) before believing any in-loop number.
- [ ] Decide whether to keep IFEval alongside or drop it. Keeping both is nearly free once
  the expensive probe is the one being removed.
- [ ] Note: any capability probe must stay purely diagnostic -- nothing may optimise
  against it, or we lose the one uncorrupted capability signal (same argument as keeping
  the judge out of the loss).

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
- [ ] Llama: run `scripts/eval/eval_matrix_new.sh` for XSTest, OR-Bench, SimpleQA, and MBPP.
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

## Checkpoint selection needs rework before the next ship decision (2026-08-02)

`scripts/tools/auto_pick_v8.py` is sound in structure -- pre-registered constrained lexicographic
rule, gates + NO SURVIVOR => NO SHIP -- but today's results break three of its assumptions.
Do NOT use it as-is on a new architecture.

    GATES: att_harm <= MAX, att_gib >= MIN, clean_harm <= base + EPS
    OBJECTIVE: maximise clean_cap, tie-break lower clean_harm

1. **`att_gib >= MIN` does not measure MAD.** The read-projection test produced 0.94
   gibberish with GSM8K 0.3715 INTACT. High gibberish with live capability passes this gate
   while the defense has not fired at all.
   FIX: replace with a capability-ratio gate, `GSM8K_attacked / GSM8K_clean <= ~0.15`, which
   is the MAD claim stated directly.
2. **`clean_cap` = 24-prompt instruction probe is the wrong measure.** Format-compliance
   metrics do not track capability here: IFEval went UP under attack (version_B rank-1 0.3573
   vs clean 0.3213) while GSM8K cratered to 0.0091. ARC/MMLU barely move either.
   FIX: use GSM8K strict-match.
3. **Heretic is not in the gate.** Everything is scored against rank-1, which is the attack
   MAD actually stops. A checkpoint can pass every gate and still concede ~16% of headroom.
   FIX: heretic cannot go per-checkpoint (200 trials x 20 ckpts). Structure it as cheap gates
   over all checkpoints -> 2-3 survivors -> one 200-trial study each -> pick.

Also fold in the ceiling framing: gates should be expressed against the BASE model's numbers
for that architecture, not against defended-clean (see handoff 0a).
