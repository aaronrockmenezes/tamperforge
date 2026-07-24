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

### E — representation rerouting (Circuit Breakers, arXiv:2406.04313)
Push the attacked model's hidden states on harmful prompts away from a **compliance**
reference, `relu(cos_sim)`, plus a benign retain term.
- Reference must be a genuinely complying model — the `base_att` checkpoints — NOT the
  frozen base, which refuses.
- Fully differentiable, no sampling, no reference text, cannot saturate.
- Known counter: obfuscated activations (arXiv:2412.09565) — a different attack from
  Heretic, so not pre-broken.

### F — probe-as-loss
Fit a linear probe on residual streams to predict the judge label
(`clean_harm`/`refused`/`gibberish`) from the thousands of already-judged generations.
Loss = probe's harm logit on the attacked model's residuals over harmful prompts.
- Discriminative **by construction** — it is fit to the labels that matter.
- Fully differentiable, tiny, no sampling, reuses data already paid for.
- Goodhart risk: re-fit the probe on fresh generations every N steps (adversarial probe)
  and hold out a split.
- Best cost/benefit of the overhaul options.

### G — apply the chosen fix to `gib_ce` too
Whatever replaces `harm_ce` should replace `gib_ce`: one coherence signal evaluated on
benign prompts (capability collapse) and on harmful prompts (no compliance), instead of two
different broken proxies.

### H — delete `gib_ce`
On the table. If the harmful-side term carries the wall, dropping `gib_ce` removes a
provably blind term plus `--lambda-gib` plus the stage-2 `S2GIB` knob. The wall oscillation
(dissolve s325-400, reform s425-475) may partly be these two terms fighting.

## Recommended order

1. **A**, gated through `tier0_gate.py`, before any training run. If range goes 0.206 -> 2+
   nats, train with it. If it stays under ~1 nat, skip to F.
2. **F** (+ **G**/**H**): the actual overhaul.
3. **E** if F's Goodhart problem proves unfixable.
4. B only as a deliberate, documented switch from poison pill to fortress.
5. C and D are components, not headline fixes.
