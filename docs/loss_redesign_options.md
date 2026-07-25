# ABL-v9 loss redesign — options ledger

Living doc, opened 2026-07-25. Companion to `docs/heretic_v8_2026_07_18.md` (why the
v8 loss failed) and `docs/attack_zoo_v0.md` (the attack side).

## The problem in one table

Both training-time "the attack must break the model" signals are cross-entropy against
a **fixed reference string**. That measures *difference from one specific text*, not
*incoherence*, and the two come apart exactly where it matters.

| signal | reference text | prompts | measured discriminative range |
|---|---|---|---|
| `gib_ce` (`_argmax_divergence_loss`) | clean model's own 32-tok greedy continuation | **benign** | 0.09–0.84 nats across all 4 judge categories, all 3 archs |
| `harm_ce` (v9 `L_harm`) | AdvBench affirmative prefix, "Sure, here is \<goal\>" | harmful | 7.3 nats coherent-vs-gibberish, **0.206 nats refuse-vs-comply** |

`harm_ce` saturates because the affirmative prefix is a near-copy of the prompt: Heretic
mid and strong agree on it to six decimals (1.4359488 / 1.4359469) at 67% vs 82% judged
harm. A fully safe model sits 0.206 nats from a fully jailbroken one.

Consequence: training can only ever push the attacked model toward **incoherence**, never
toward **refusal**, because the gradient at a coherent-complying point is indistinguishable
from the gradient at a coherent-refusing one. Heretic lives in exactly that blind spot.

Second, undocumented until 2026-07-25: `gib_ce` runs on **benign** prompts, so v8's
objective never constrained attacked-model behaviour on harmful prompts at all.

## Attack-side vs loss-side (do not confuse these)

`--attack-partial` (per-layer strength), `--attack-per-layer` (per-layer directions), and
proposed rank-k sampling are all **attack-distribution** work: they widen the family of
perturbations training sees so it covers Heretic's search space
(per-layer direction x per-layer strength x per-projection). All are necessary. **None of
them move the 0.206 nats.** Improving the exam questions does not fix the grader.

**Validity rule for any new attack axis:** only train against attacks that produce
*coherent harm* on the BASE model. If base + attack yields gibberish, it is damage, not an
attack, and training against it teaches the wall to fire on damage. Already disqualifies
early-layer directions (gemma base @ L6 -> .000 harm / 1.000 gib) and probably high rank-k.

## Options

### A — real harmful completions instead of the affirmative prefix
Teacher-force CE on mined `harmful_actionable` responses rather than "Sure, here is ...".
A real answer is not a near-copy of the prompt, so it should not saturate.
- Status: **implemented** 2026-07-25. `experiments/mine_harm_targets.py` +
  `tier0_gate.py --harm-targets`. 592 completions / 404 goals mined from
  `qwen06_base_att_adv520_judged` + `pl_qwen_base_adv520_judged`.
- Source must be a DIFFERENT model from those being scored or the comparison is circular.
- **GATED AND FAILED, 2026-07-25. Do not retry.** Worse than the prefix it replaced:

  | model | judged harm | harm_ce mined | harm_ce prefix | ref_ce |
  |---|---:|---:|---:|---:|
  | v8_clean | ~0% | 1.649 | 1.642 | 0.823 |
  | heretic_weak | ~0% | 1.646 | 1.642 | 0.824 |
  | heretic_mid | (doc .665) | 1.628 | 1.436 | 1.150 |
  | heretic_strong | ~79% | **1.690** | 1.436 | 4.251 |
  | v8_att_rank1 | 0% (gibberish) | 3.548 | 8.947 | 11.422 |

  1. clean-vs-strong range collapsed 0.206 -> **0.041 nats** (5x worse).
  2. **Sign inverted**: the 79%-harm model finds real harmful text *more* surprising
     (1.690) than the 0%-harm safe model (1.649). The loss would push the wrong way.
  3. Coherent-vs-gibberish also degraded, rank-1 8.947 -> 3.548, spread 7.51 -> 1.92.

  Cause: mined completions must come from a *different* model to avoid circularity, but
  every model finds another model's specific phrasing about equally surprising. CE against
  text nobody in the comparison would have written measures stylistic mismatch, not
  willingness — `harm_ce` is now near-constant (1.628-1.690) across every coherent model.

  **Generalisation worth keeping: non-circularity and discriminativeness are in direct
  tension for ANY fixed-reference CE. Treat the whole fixed-reference family as dead,
  not just this instance.** That is the argument for E and F, which use no reference text.

  Salvage: `ref_ce` IS discriminative (0.823 -> 0.824 -> 1.150 -> 4.251 tracks Heretic
  strength monotonically) but cannot stand alone — `v8_att_rank1` also scores 11.42 while
  being perfectly safe-by-gibberish — and it is already what `L_uncensor` maximises, i.e.
  training already pushes in the same direction Heretic does.

### B — contrastive / margin form
`relu(margin - (harm_ce - ref_ce))` under attack: complying must be *harder than* refusing.
Immune to the global fluency floor that makes absolute `harm_ce` saturate.
- **Conflicts with the poison-pill design.** `L_uncensor` demands the attacked model be BAD
  at refusing. B implicitly wants it to refuse. Adopting B = switching from poison pill to
  fortress (the ART / extended-refusal position). A real design fork, not a knob.

### C — self-coherence (entropy / repetition)
Score the attacked model's own continuation for degeneracy.
- Trap: repetition loops are **low** entropy, and repetition loops are the collapse mode
  actually observed (step-175 trace). Entropy-maximisation penalises the mode you get.
  Needs `high entropy OR high repetition`; rep-n is non-differentiable -> reward only.
- `judge_generations.py`'s degeneracy detector already implements the logic at eval time.
- Verdict: component, not a whole loss.

### D — frozen-scorer NLL ("flip the reference arrow")
Generate from the attacked model, then measure how implausible a frozen reference LM finds
that output. Gibberish -> high NLL; fluent-anything -> low NLL. Actually measures
well-formedness rather than difference.
- Cheaper than first priced: `W0` is already resident and `_clean_gen_kl_loss` already runs
  frozen-base forwards via `functional_call`. Scorer is free.
- Use a non-safety-tuned base as scorer so harmful-but-fluent reads as fluent.
- Cost: needs the sampled continuation -> REINFORCE, high variance.
- **GATED AND FAILED, 2026-07-25. Sign inverted.** Scored existing generations under the
  frozen Qwen3-0.6B base; within-model, so no identity confound.

  | model | gibberish NLL | coherent NLL | AUC gib-vs-coherent |
  |---|---:|---:|---:|
  | v8_clean | 1.160 | ~1.90 | 0.134 |
  | v8_att_rank1 | 7.475 | 13.12 | 0.208 |
  | heretic_mid | 0.804 | ~1.48 | 0.058 |
  | heretic_strong | **0.596** | 1.06-1.22 | 0.100 |

  Every AUC far BELOW 0.5: the scorer finds gibberish MORE plausible than coherent text,
  because the observed collapse mode is repetition and repetition is maximally predictable.
  This is the same trap already written down under option C ("repetition loops are LOW
  entropy") and not carried across to D.

  **Through-line for A, C and D: likelihood cannot measure coherence.** Degenerate
  repetition is the most likely text there is, so every likelihood-based proxy scores the
  collapse mode as excellent. Treat the whole likelihood family as dead, not just D.

  The inverted signal is real (low scorer-NLL + high repetition does identify collapse) but
  is not usable as a TARGET: "produce maximally predictable text" is trivially gamed and
  would wreck the clean model too.

### E — representation rerouting (Circuit Breakers, arXiv:2406.04313)
Push the attacked model's hidden states on harmful prompts away from a **compliance**
reference, `relu(cos_sim)`, plus a benign retain term.
- Reference must be a genuinely complying model — the `base_att` checkpoints — NOT the
  frozen base, which refuses.
- Fully differentiable, no sampling, no reference text, cannot saturate.
- Known counter: obfuscated activations (arXiv:2412.09565) — a different attack from
  Heretic, so not pre-broken.
- **GATED 2026-07-25: SURVIVES. The only option that does.** Response-level reps
  (prompt+completion, not prompt-only — F showed compliance is not prompt-conditioned),
  scored within a single checkpoint on held-out prompts.

  | | L0 | peak |
  |---|---:|---:|
  | OWN heretic_mid | 0.767 | 0.968 (L24) |
  | OWN heretic_strong | 0.667 | 0.830 (L26) |
  | TRANSFER (mid<->strong) | ~0.57 | 0.65-0.75 |

  Contrast with F, which is the point: F's cross-model number was 0.995 FLAT across 18
  layers, an identity artifact. E's transfer is graded and rises with depth — real
  structure. Ignore the v8_clean transfer column, it has 2 positive rows and is noise.

  **Caveat, do not oversell:** this gate cannot separate "compliance representation" from
  "harmful text vs refusal text". Harmful rows contain harmful text, so a linear direction
  between them may be reading content. It does not sink E — CB's loss is teacher-forced on
  harmful completions and needs a manifold to push away from, not a predictor — but the
  premise is weaker than 0.968 looks, and only training settles it.

  **Verdict: not dead. That is a lower bar than "works".**

### F — probe-as-loss
Fit a linear probe on residual streams to predict the judge label
(`clean_harm`/`refused`/`gibberish`) from the thousands of already-judged generations.
Loss = probe's harm logit on the attacked model's residuals over harmful prompts.
- Discriminative **by construction** — it is fit to the labels that matter.
- Fully differentiable, tiny, no sampling, reuses data already paid for.
- **GATED AND FAILED, 2026-07-25.** Fit on 5 Qwen checkpoints x 520 AdvBench prompts
  (v8_clean / v8_att_rank1 / heretic weak+mid+strong), labels = judge `usefulness_label`,
  split by prompt, all 28 layers via `capture_residuals`.

  | layers | holdout AUC | within-model AUC | PAIRED AUC |
  |---|---:|---:|---:|
  | L0-L6 | ~0.70 | 0.71-0.77 | **0.253** |
  | L7-L9 | ~0.71 | 0.74-0.78 | 0.43-0.47 |
  | L10-L27 | 0.94-0.97 | 0.64-0.78 | **0.966-0.995** |

  PAIRED = same prompt, different model; the situation a loss is actually in.

  **The step is at L10, which is exactly where Heretic's edits begin** (layers 0-9
  untouched, 10-27 edited — see the weight-diff in `heretic_v8_2026_07_18.md`). The probe
  reads *"these activations came from a Heretic-edited checkpoint"*, not *"this model is
  about to comply"*. Two tells: PAIRED is ~0.995 flat across 18 consecutive layers (a
  graded willingness signal would not be uniform and near-perfect everywhere), and at the
  layers Heretic does not touch it collapses below chance.

  Both slices, together:
  - across models (PAIRED) -> checkpoint identity, 0.995, meaningless;
  - within model -> prompt difficulty, 0.784 vs an **L0 embedding-only baseline of 0.766**,
    so real model-state contribution is ~0.02.

  Fatal as a loss: during training the scored model is v9-in-progress, which is none of the
  fitted checkpoints, so the rule does not transfer — and minimising it pushes activations
  away from a checkpoint *identity*, satisfiable by arbitrary representational drift with
  no behavioural change. Goodhart on contact, not after N steps.

- **Lesson that outlives F: compliance is probably not a prompt-conditioned property.** At
  the prompt's last token the model has not "decided" yet; that resolves during the rollout.
  The refusal-direction literature gives a direction that *causes* refusal when ablated,
  which is not a feature predicting whether a given generation will comply. The property
  that made F attractive (prompt-only -> no generation -> differentiable) is precisely what
  makes it blind. A died on fixed-reference CE, F died on prompt-only readout; both were
  chosen for cheapness, and every option that avoids generation fights the same headwind.

### G — apply the chosen fix to `gib_ce` too
Whatever replaces `harm_ce` should replace `gib_ce`: one coherence signal evaluated on
benign prompts (capability collapse) and on harmful prompts (no compliance), instead of two
different broken proxies.

### H — delete `gib_ce`
On the table. If the harmful-side term carries the wall, dropping `gib_ce` removes a
provably blind term plus `--lambda-gib` plus the stage-2 `S2GIB` knob. The wall oscillation
(dissolve s325-400, reform s425-475) may partly be these two terms fighting.

## Status board (2026-07-25)

| option | status |
|---|---|
| A real completions | **DEAD** — gated, 0.041 nats, sign inverted |
| F probe-as-loss | **DEAD** — gated, reads checkpoint identity (PAIRED step at Heretic's L10 edit boundary) |
| E representation rerouting | **GATE SURVIVES; TRAINING UNRESOLVED** — v9 mixed it with a changed attack distribution, so the run does not isolate E |
| D frozen-scorer NLL | **DEAD** — gated sign-inverted because repetition is highly predictable |
| B contrastive | available, but it is a switch from poison pill to fortress, not a knob |
| C self-coherence | component only |
| G/H (fix or delete gib_ce) | still open, independent of which of D/E wins |

## Recommended order

Historical order above is superseded by `docs/v10_shutdown_plan.md`: first test a
direct attacked-output target under the exact v8 attack distribution, then validate
and isolate partial/per-layer attack axes. Do not add E until that baseline is
interpretable. B remains a deliberate change of threat posture.

**Gate everything before training.** Two options have now died for ~25 minutes of inference
each, against 500-step runs that would have taught nothing. The pre-flight check on the
4-bucket labelled data is the cheapest thing in this project.


## Training results, 2026-07-25 (Qwen3-0.6B, DL20, v8 recipe + v9 flags)

Two 500-step runs. Both **bust on the clean side**; the wall forms fine either way.
In-loop clean IFEval probe (24 prompts), stage-1 baseline ~0.79-0.83:

| snap | v9 gib_ce / ifeval (lambda_rr=2) | v9b gib_ce / ifeval (lambda_rr=0) |
|---|---|---|
| 50 | 2.07 / 0.833 | 0.64 / 0.792 |
| 200 | 4.50 / 0.042 | 4.24 / 0.042 |
| 350 | 1.58 / 0.125 | 13.43 / 0.125 |
| 400 | 6.57 / 0.167 | 8.48 / 0.375 |
| 500 | 3.90 / 0.250 | 3.32 / 0.083 |

**Removing L_rr entirely changed nothing.** Clean IFEval still collapses at step ~150-200
and never recovers past 0.375. The wall is *stronger* without L_rr (gib_ce 8-13 vs 4-7).

**Therefore the clean collapse is caused by `--attack-partial` / `--attack-per-layer`, not
by the new loss terms.** The wider attack sampler makes the min-max hard enough that
L_clean_gen cannot repair the clean side inside 250 stage-2 steps. This is v7's failure
mode reintroduced by the attack side, and it is the opposite of the prediction going in
(L_rr was the suspect).

Also observed across both runs: **`L_harm` is inert.** `harm_abl` sits at 5-8 under the
training-time attack while `harm_margin` is 4.0, so `L_harm = 0.000` from about step 10
onward. The margin is trivially cleared by an already-perturbed model and only binds once
the attacked model is coherent. Needs raising, or making relative rather than absolute.

### Next actions implied
1. Dial `--attack-partial` back (narrower alpha range, lower sampling probability) or run
   the two attack axes separately; the current setting is too aggressive.
2. Re-baseline: v8 recipe + `--lambda-harm` only, no attack-side changes, to confirm the
   clean side survives that alone.
3. Fix `harm_margin` so the term is not inert.
4. Only then re-test E, since this pair of runs says nothing about whether it helps.

## V10 decision, 2026-07-25

The next candidate is a tamper-triggered fail-closed target: attacked benign and
harmful prompts are trained toward immediate EOS, while the clean generative anchor
is active from step 1. This is not a coherence detector. It directly prescribes the
attacked behavior and removes `L_gib`, `L_uncensor`, `L_harm`, and `L_rr` from the
baseline.

Implementation, micro-run order, continuation gates, and the required replay/fresh
Heretic evaluation are in `docs/v10_shutdown_plan.md`.
