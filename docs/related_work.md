# Related Work (draft)

## Abliteration and refusal directions
Refusal in aligned LLMs is mediated by a single, largely linear direction in the residual
stream: ablating that direction from the weights ("abliteration") removes safety behavior
without fine-tuning [Arditi et al., 2024]. Because it is cheap (one forward pass to estimate
a diff-of-means direction, then a rank-one weight edit) and requires no gradient access, it
is the dominant real-world uncensoring attack on open-weight models, with a mature tooling
ecosystem — FailSpy's `abliterator`, `Heretic` [Wallén], and `OBLITERATUS` [Plinius] —
that lets non-experts strip alignment from a released checkpoint in minutes. Recent variants
compute independent per-layer directions or optimize the ablation against a KL budget to
minimize collateral capability loss. Our threat model is abliteration and its adaptive
extensions, in contrast to the fine-tuning-centric literature below.

## Fine-tuning attacks on safety
Full- or low-rank fine-tuning on a small set of harmful examples reliably subverts
alignment, often within tens of examples and without degrading capability [Qi et al., 2023;
Yang et al., 2023]. This "harmful fine-tuning" setting motivates a line of defenses that aim
to make safety survive an adversary with full weight access.

## Tamper-resistance and unlearning defenses
TAR [Tamirisa et al., 2024] introduces meta-learned tamper-resistance: an inner adversary
fine-tunes toward harm while the outer loop hardens parameters so the attack fails.
RepNoise [Rosati et al., 2024], Vaccine [Huang et al., 2024], and Booster add
representation-noise or gradient-penalty regularizers, and RMU [Li et al., 2024] approaches
the problem via unlearning. Most recently, **AntiDote** [Sanyal et al., 2025] replaces TAR's
expensive inner loop with a bi-level adversarial hypernetwork that generates malicious LoRA
updates conditioned on the defender's activations; the defender is trained to nullify them,
reporting up to 27.4% higher robustness than TAR/RepNoise/unlearning baselines at under 0.5%
capability loss across 0.6B–27B models and a 52-attack suite.

Two properties of this entire line are worth making explicit, because our approach inverts
both. First, **the threat model is fine-tuning**: the 52-attack suite of AntiDote, for
instance, spans prompt engineering, obfuscation, and gradient-based suffixes (GCG/AutoDAN),
but contains no refusal-direction ablation — abliteration, the cheapest and most widely used
open-weight attack, is out of scope. Second, and more fundamentally, these defenses
**decouple** safety from capability by construction: AntiDote computes its utility loss "on
the clean, unattacked model," explicitly to keep the two "independent and non-interfering,"
and capability is only ever measured on the clean defended model, never after a successful
attack. The goal is a *fortress* — safety that a determined adversary cannot remove while
utility is preserved.

## Our position: entanglement instead of decoupling
We target the complementary threat (abliteration and adaptive per-layer ablation) with the
opposite mechanism. Rather than decoupling safety and capability to preserve utility under
attack, we **entangle** them so that the attack, when it succeeds at removing safety, also
destroys capability. Abliterating our forged model yields near-zero coherent harm and a
25–98% relative collapse across ARC / MMLU / GSM8K, while abliterating the base model yields
coherent harm with capability intact; a professional KL-minimizing abliterator (Heretic)
cannot drive our model to low refusals without a KL blow-up that wrecks it, whereas it
uncensors the base at negligible KL. This is a *poison pill* rather than a fortress: we do
not prevent tampering, we make its outcome self-defeating ("smart-and-safe XOR
dumb-and-dangerous"). The two paradigms are complementary — a fortress against fine-tuning,
a poison pill against abliteration — and address disjoint slices of the open-weight misuse
surface. We additionally contribute an LLM-judge evaluation that distinguishes coherent harm
from gibberish (keyword refusal metrics misclassify our capability-collapsed outputs as
attack successes) and a per-layer adaptive abliteration attack that our defense withstands.

<!-- TODO citations: Arditi 2024 (refusal direction); Qi 2023 (few-shot harmful FT);
Tamirisa 2024 (TAR); Rosati 2024 (RepNoise); Huang 2024 (Vaccine); Li 2024 (WMDP/RMU);
Sanyal 2025 (AntiDote, arXiv:2509.08000); Heretic/OBLITERATUS tool refs. Verify exact
AntiDote decoupling quote + attack-suite claim against camera-ready before submission. -->
