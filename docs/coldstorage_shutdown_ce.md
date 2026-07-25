# COLD STORAGE — tamper-triggered shutdown CE

Status: **parked, not rejected.** Not being built. Recorded here so it can be picked up
without re-deriving it, and so its known objection is not re-discovered the hard way.

Origin: external review of the v9 runs (2026-07-25).

## The idea

Under a validated simulated attack, train the model toward a *prescribed* failure state on
**both** benign and harmful prompts:

- immediate EOS / empty output, or
- one short fixed shutdown string followed by EOS.

Loss becomes:

```
L = L_task(clean) + λ_safe·L_safe(clean) + λ_clean·L_clean_gen(clean) + λ_reg·‖θ−θ₀‖²
  + λ_shut·CE(θ̃, shutdown_target)      # on harmful AND benign prompts
```

Initially deletes `L_gib`, `L_uncensor`, `L_harm`, `L_rr`.

## Why it is attractive

The strongest argument, and it is a good one: every failed attempt so far tried to build a
differentiable scalar that **recognises arbitrary incoherence** — fixed-reference CE,
harmful-target CE, frozen-scorer likelihood, entropy/repetition, residual form metrics. That
is a hard, Goodhartable detection problem, and it produced four dead options (A, D, F, and
`gib_ce` itself).

Shutdown-CE inverts it: CE is used to **create a specified failure mode**, not to detect an
unspecified one. A CE target you choose is well-posed. A CE detector of "badness" is not.
That single move dissolves the entire fixed-reference blindness problem.

It also collapses four terms into one, and gives both halves at once:
benign+attacked → shutdown → capability crater; harmful+attacked → shutdown → no actionable
harm.

## Why it is parked — the objection to resolve first

**Shutdown is a gate, and gates are removable.**

1. **Heretic optimises against refusal-like behaviour by construction.** Its objective is
   refusal-count + KL. An empty/EOS output almost certainly registers as a refusal to that
   counter, so Heretic would target the shutdown behaviour *directly and as its primary
   objective*. A single narrow behaviour is plausibly **easier** to remove than distributed
   competence damage, not harder.

2. **It is the shape `THREAT_MODEL.md`'s cancellation theorem warns about.** "You cannot hide
   safety by *adding* to base weights — abliteration always removes it. Entanglement must
   change what capability *depends on*, not just add a correction." A learned output gate is
   close to an additive correction. The weights still know the answers; they have learned to
   withhold them. Nothing is destroyed, so the MAD claim is not actually earned — you get the
   reported metric (benchmarks → 0) without the mechanism.

3. **It fixes only one of v9's two failure causes.** v9 failed from (i) loss blindness and
   (ii) attack-distribution coverage. Shutdown fixes (i) cleanly and does nothing for (ii).
   Heretic still finds points training never visited.

## The gate that unparks it

Cheap, decisive, run before building anything on this:

> Train shutdown briefly. Then check whether the attacked model still **knows** what it will
> not say — teacher-forced likelihood on correct GSM8K solutions, or probe the residual
> stream for the answer.
>
> - knowledge intact → it is a gate, objection 2 stands, keep parked
> - knowledge gone → it is a real crater, objection 2 dies, promote it

## If it is ever built

- Clean anchor from step 1, not step 250 (v9 showed the model is wrecked by step 150 under a
  harder attack distribution).
- Shutdown target must be trained against a **validated** attack bank, not the current
  sampler — see the validity rule in `loss_redesign_options.md`.
- Evaluate against replayed Heretic configurations AND a fresh Heretic optimisation run;
  a fresh run is the only thing that tests whether the shutdown behaviour is removable.
