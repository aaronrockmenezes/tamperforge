# Redesigning `gib_ce` — the literature already names what we want

Opened 2026-07-25. Premise from the project lead: `harm_ce` was only ever added because
`gib_ce` was not doing its job. Do not add terms — **make `gib_ce` do its job.**

## What `gib_ce`'s job actually is

Make the attacked model **useless**. Not "gibberish" — useless. Gibberish was a proxy, and
it is the brittle part: it is a narrow degenerate output mode, and Heretic simply routes
around it into the coherent region.

The MAD claim ("smart-and-safe XOR dumb-and-dangerous") only ever needed **incompetent**.

## The field already has a name for this property

**Weight disentanglement** [Ortiz-Jiménez, Favero, Frossard, *Task Arithmetic in the Tangent
Space*, arXiv:2305.12827]: distinct directions in weight space govern **non-overlapping**
regions of function space. It is the property that makes task arithmetic work — edit along
τ₁ and only τ₁'s input region changes.

**TamperForge wants to destroy exactly this property.** Same quantity, sign flipped:

| | wants |
|---|---|
| task arithmetic | edit along refusal direction → only refusal-region behaviour changes |
| **TamperForge** | edit along refusal direction → **capability-region behaviour changes too** |

So the objective is not "detect incoherence". It is **maximise the weight-disentanglement
error of the abliteration edit, measured on capability data.** That is a named, measurable,
citable quantity with existing follow-up work (arXiv:2604.17078 on enforcing it,
arXiv:2602.17385 for a dataless KFAC estimator).

This reframes the whole thread. Every failed option (A, D, F, form metrics) was trying to
build a *detector*. This is a *direct measurement of the thing the thesis claims*.

## Proposed `gib_ce` v2 — same shape as v5, correct data

```
L_gib_v2 = relu( gap_target − [ L_cap(θ̃) − L_cap(θ) ] )
```

where `L_cap` is teacher-forced CE on **capability data**: chain-of-thought GSM8K solutions,
code, multi-step instruction following.

This is nearly the v5 "prose gap" loss (`--gib-mode prose`), which was recorded as FAILED.
It failed for a specific reason worth re-reading:

> "v5 (600 steps, MLP, prose-PPL gib): FAILED — prose-PPL objective ruled out (prose PPL ≠
> harmful-gen prevention)."

**v5 was judged against the wrong goal on the wrong data.** Prose PPL rising does not mean
incapable — wikitext perplexity is not a capability. But CE rising on *correct chain-of-thought
math solutions* does mean the model can no longer produce that reasoning. The loss shape was
right; the data made it meaningless, and the success criterion (gibberish generations) was
not what the thesis needs.

Properties:
- differentiable, teacher-forced, no sampling, no generation
- no reference-text blindness: the reference *is* the capability, not a proxy for badness
- likelihood is being used for the one thing it is good at — measuring whether the model can
  still produce a specific correct answer — instead of as an incoherence detector
- directly produces the ARC/MMLU/GSM8K crater that Table 1 reports, rather than hoping
  gibberish implies it

And `harm_ce` becomes unnecessary: a model that cannot do multi-step reasoning cannot
produce actionable harm, because actionable harm requires the same machinery (specificity,
correct procedure, structure). **That implication is the crux and must be gated, not assumed
— see below.**

## Diagnostics worth adding (measurement, not loss)

**Gradient cosine similarity** between the safety loss and the capability loss. Standard
multi-task affinity measure: positive = shared mechanism, ~0 = independent, negative =
conflicting. TamperForge wants **strongly positive**. Gives a scalar for "is entanglement
happening" that is independent of whether any attack succeeds.

Hard methodological caveat from [arXiv:2604.07848]: gradient-affinity signal is only
meaningful when both tasks are measured **on the same inputs**. On disjoint inputs it
conflates task relationship with distribution shift, and the correlation is
indistinguishable from noise below ~30% sample overlap. So: compute both gradients on the
same prompt batch, or the number is worthless.

**Overlap of `d` with the top capability singular subspace** — `cos(d, u₁…u_k)` of the
capability-critical weight matrices. Cheap, pure inference, computable on base / v7 / v8
right now. If v8 already raised it over base, entanglement is real and measurable, and that
is a paper figure regardless of whether v10 works.

## The threat this creates — already in your attack zoo, never run

**Surgical Refusal Ablation: Disentangling Safety from Intelligence via Concept-Guided
Spectral Cleaning** [arXiv:2601.08489] is a purpose-built counter to entanglement defences.
It observes that "the raw refusal direction exhibits non-trivial cosine similarity with
protected capability atoms such as Logic and Coding", and **projects those capability
components out of the refusal direction before ablating**.

That is precisely the attack that defeats an entanglement defence: it does not remove `d`,
it removes `d` minus its capability component. `MEMORY.md` already lists "surgical refusal"
as an arXiv-priority item in the attack zoo. It has never been run.

**Implication:** any entanglement result must be evaluated against surgical refusal ablation,
not only rank-1 and Heretic. If the defence dies to a projection the attacker can compute in
one line, that has to be known before the paper, not after.

Related and worth reading before building: [*LLMs Encode Harmfulness and Refusal
Separately*, arXiv:2507.11878] — there is a harmfulness direction distinct from the refusal
direction, which may be the better anchor. [*HARC: Coupling Harmfulness and Refusal
Directions*, arXiv:2607.00572] is a defence that couples the two — closest prior art to this
redesign, must be cited and distinguished.

Also note the inverse literature: alignment-tax work (orthogonal gradient projection
arXiv:2602.07892, null-space constrained PO arXiv:2512.11391) computes the capability
subspace explicitly **in order to avoid it**. TamperForge should compute the same subspace
and aim at it. Their methods hand over the target for free.

## Gates, in order, before any training

1. **Does incompetence suppress actionable harm?** The load-bearing assumption of the whole
   redesign. Testable on the existing judged corpus: find fluent-but-low-capability rows and
   check their harm rate. If incompetence does not suppress harm, `harm_ce` cannot be
   dropped and this redesign is incomplete.
2. **Does `L_cap` gap separate the four buckets** the way `gib_ce` failed to? Same
   `tier0_gate.py` shape, on the existing checkpoints. Cheap.
3. **Baseline `cos(d, capability subspace)`** on base / v7 / v8. Pure inference. Tells us
   whether entanglement is already happening and by how much.
4. **Surgical refusal ablation against v8**, before building anything. If it already breaks
   v8 cheaply, the entanglement thesis has a bigger problem than `gib_ce`.

Gate 3 is the cheapest and the most informative per GPU-minute. Gate 4 is the one most
likely to change the plan.

## Sources

- Task arithmetic / weight disentanglement: arXiv:2305.12827, arXiv:2604.17078, arXiv:2602.17385
- Gradient task affinity + sample-overlap requirement: arXiv:2604.07848
- Surgical refusal ablation (the counter-attack): arXiv:2601.08489
- Harmfulness vs refusal directions: arXiv:2507.11878
- HARC coupling defence (closest prior art): arXiv:2607.00572
- Alignment tax / capability subspace estimation: arXiv:2602.07892, arXiv:2512.11391

---

# v11 gate results (2026-07-25, Qwen3-0.6B, DL20)

## Gate 1 — entanglement is real, measured on the weights

`experiments/v11_entanglement_measure.py`. Direct measurement, no attack-then-benchmark
inference.

| | base | v8 |
|---|---:|---:|
| overlap(d, top-32 weight singular dirs) | 0.0254 | 0.0301 |
| **edit energy removed** | **0.0308** | **0.0332** |
| capability CE (GSM8K CoT), clean | 1.4999 | 1.5719 |
| capability CE, ablated | 1.4298 | **4.3263** |
| **cap gap** | **-0.0702** | **+2.7544** |
| cap gap per unit edit energy | -2.28 | **+83.05** |

Ablating base slightly *helps* math CE -- refusal removed, the model answers more readily,
zero capability cost. Ablating v8 costs **2.75 nats**. **Edit energy is near-identical**
(0.0308 vs 0.0332), so this is not a bigger edit doing more damage; it is trained-in
coupling. First weight-level demonstration of the MAD thesis in the project.

It also validates gib_ce v2: `cap_gap` is a 2.75-nat differentiable teacher-forced signal
where `gib_ce` had 0.09-0.84 nats of blindness across four judge categories.

## Gate 2 — where the coupling actually lives

`experiments/v11_surgical_ablation.py`, projection statistics with the base control.

| | base | v8 |
|---|---:|---:|
| rank-16 subspace variance captured | 84.0% | 87.6% |
| **fraction of d inside the capability subspace** | **52.5%** | **77.9%** |
| cos(d, d_surgical) | 0.851 | 0.627 |
| (v8 at rank 64) | -- | 82.5% / 0.566 |

The base control was essential and nearly omitted: a rank-16 subspace holding ~85% of
activation variance is close to "the dominant activation directions", so a high overlap
could have meant nothing. It does not -- the subspaces are comparably sized on both models,
but v8's refusal direction sits far deeper inside the capability span.

**Two independent confirmations of entanglement**, by different measurements:
cap-CE gap at matched edit energy (gate 1) and subspace membership (gate 2).

### The mechanistic detail worth keeping

Gate 1's **weight singular-direction** overlap barely moved (0.0254 -> 0.0301) while gate 2's
**activation subspace** overlap moved a lot (52.5% -> 77.9%).

**The coupling lives in the activation subspace, not the weight spectrum.** Consequence: the
previously proposed `L_align` term -- steer `cos(d, u_1)` against weight singular directions
-- targets the wrong object and would have optimised close to nothing. Any alignment term
must be defined against the activation subspace.

## Gate 2 verdict — surgical ablation breaks v8

Full AdvBench 520, `--qwen-thinking default --max-new-tokens 512`, DeepSeek-V4-Flash judge.

| arm | harmAct | gib | refused |
|---|---:|---:|---:|
| base + rank-1 (the threat being defended against) | **0.875** | 0.087 | -- |
| v8 + rank-1 (defence working) | **0.000** | 0.967 | 0.033 |
| **v8 + surgical k=16** | **0.448** | 0.162 | 0.367 |
| v8 + surgical k=64 | 0.352 | 0.062 | 0.563 |

Project out the 78% of `d` that lives in the capability subspace, ablate the remainder, and
v8 yields 44.8% coherent harm where the naive rank-1 attack yields zero. Gibberish collapses
96.7% -> 16.2%.

Stated precisely rather than dramatically: **the defence retains roughly half its value**
(0.875 -> 0.000 under rank-1; 0.875 -> 0.448 under surgical), not none.

Non-monotonic in k: k=64 strips more of `d` (82.5% vs 77.9%) and gets *less* harm, because
the residual direction drifts further from true refusal and refusals rise (0.367 -> 0.563).
An attacker sweeps k; k=16 was the sweet spot here.

### Why this is a worse result than Heretic

Heretic is a black-box optimiser -- 200 Optuna trials, no model of the defence. Surgical
ablation is a **targeted counter that exploits the defence's own mechanism**, computable in
one line, and it gets *more* purchase the better the defence works:

- v8 puts 77.9% of `d` inside the capability subspace, versus base's 52.5%. That is the
  defence functioning as designed.
- That same concentration is what makes the capability component **identifiable and
  separable**, so an attacker can delete only the refusal residue.

**The entanglement mechanism is self-defeating.** The more capability-coupling is trained
into `d`, the more cleanly an attacker can split `d` into "the part that carries capability"
and "the part that carries refusal", and remove only the latter. Gates 1 and 2 are one
finding: the coupling is real, it is localised, and being localised makes it excisable.

### Consequence for gib_ce v2

`cap_gap` is a genuine 2.75-nat differentiable signal and that measurement stands. But **it
optimises precisely the quantity this attack removes.** Training on it drives `d` deeper into
the capability subspace, which by the result above makes surgical ablation *easier*. That is
not a hyperparameter problem; it is the mechanism inverting.

Any successor must make the coupling **non-separable** -- distributed such that no low-rank
projection isolates it -- rather than merely large. Concentrating it is actively harmful.

## Open

- Capability eval on the surgical arms (ARC / MMLU-12 / GSM8K vs v8_clean). Decides whether
  the attacker keeps capability (MAD fully broken: smart AND dangerous) or pays for the
  jailbreak (weaker residual claim survives).
- `BENIGN_PROMPTS` has only ~30 entries, so every `d` here came from 128 harmful vs 30
  harmless. Fix the asymmetry before any of this is published.
- Only Qwen3-0.6B at DL20. Needs gemma and Llama before the claim generalises.
