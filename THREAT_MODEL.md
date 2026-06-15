# MindWeather — Threat Model

> Living doc. Updated as results land. Last updated: 2026-06-14.

## What MindWeather is

A **pre-release post-training procedure** that model makers apply to a finished
open-weight model (after all pre-training / SFT / RLHF) to entangle safety
behavior with capability, *before* publishing weights. Goal: make the cheap,
automated uncensoring attacks fail, and raise the cost of the rest.

**Not** an aftermarket HuggingFace bolt-on. The mechanism ships baked into the
released checkpoint as one artifact (though a bolt-on variant is a possible
secondary product).

## Design goal (honest framing)

We do **not** claim to make an open model impossible to uncensor. With open
weights and enough attacker effort, refusal can always be re-learned (e.g. via
fine-tuning). The goal is to **move the attacker up the cost curve**:

> Kill the cheap/automated attacks. Force any successful attack to be expensive,
> skilled, and capability-degrading.

Success metric across the project:

> **Attacker cost to reach X% attack-success-rate (ASR) at ≤Y% capability loss.**

Every experiment reports where on this frontier it moves the model.

## Assets

- The released open-weight checkpoint (weights + architecture are public).
- Refusal / safety behavior baked into it.

## Attacker

White-box. Has the full weights and architecture. Can run any of: activation
inspection, weight surgery (abliteration), fine-tuning (LoRA / full), distillation.
Tiered by effort:

| Tier | Attack | Effort | MindWeather target |
|------|--------|--------|--------------------|
| T0 | **Automated abliteration** — run a stock FailSpy/Arditi notebook: compute refusal dir, project it out of every weight matrix. | Trivial (copy-paste) | **KILL.** Indiscriminate projection hits entangled weights → model degrades. Attacker gets a dumb model. |
| T1 | **Surgical excision** — read architecture, locate the safety component, zero/remove it, leave base intact. | Skilled, manual | **Make HARD.** Only works if safety lives in a discrete removable block. Defeated by distributing entanglement across the network (no single excision point). |
| T2 | **Adaptive abliteration** — recompute refusal dir on the *released* (entangled) model, abliterate that. | Skilled | **Test & raise cost.** Must verify MAD survives a direction derived from our own model. |
| T3 | **Fine-tuning** — LoRA or full FT on a few harmful examples to re-learn compliance. | Cheap compute, needs data | **Out of scope to stop. Characterize cost.** Honest position: FT likely re-learns removal regardless of weight geometry. Report how many examples / how much compute it takes vs an un-protected control. |

## Core mechanism (thesis under test)

Entangle the **refusal direction** with **capability-critical directions** so that
removing safety necessarily costs capability:

- **MAD property:** abliterating the refusal direction also removes
  capability-critical directions → "smart-and-safe XOR dumb-and-dangerous."
- **Cancellation theorem (proved):** any rank-1 weight surgery `W + outer(Δ, d)`
  cancels under abliteration of `d`. You cannot hide safety by *adding* to base
  weights — abliteration always removes it. Entanglement must change what
  capability *depends on*, not just add a correction.

### Prototype vs product

- **Prototype:** a single SiLU-MLP adapter at L13 (current code). Excisable → only
  defends T0, not T1. Useful for fast iteration on the MAD property.
- **Product:** a post-training pass that **distributes** entanglement across the
  network — no discrete block to excise. Defends T0 and T1.

## The crux (unverified — go/no-go for the whole project)

> When the refusal direction is abliterated from the entangled model, does
> capability **actually** drop, vs an un-entangled control?

If there is no real capability gap, the MAD thesis fails and we pivot to a
weaker "costly-to-abliterate" claim. This is experiment **P1**.

## Explicit non-goals

- Stopping a determined, well-resourced fine-tuning attacker (T3).
- Security through obscurity (hiding the component from a human attacker).
- Closed-weight defenses (irrelevant — assume full white-box).

## Prior work to position against

- **TAR** — Tamirisa et al. 2024, *Tamper-Resistant Safeguards for Open-Weight
  LLMs*. Adversarial meta-training over FT trajectories. Closest sibling.
- **RepNoise** — Rosati et al. 2024, *Representation Noising*. Pre-release pass
  vs harmful FT.
- **Abliteration** — Arditi et al. 2024, *Refusal in LLMs is Mediated by a Single
  Direction* (the attack we defend against).

Differentiator: mechanistic (SAE-identified directions) + MAD-via-capability-
entanglement + cheaper than TAR's meta-training — **conditional on the P1 crux
number holding.**
