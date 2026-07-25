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
