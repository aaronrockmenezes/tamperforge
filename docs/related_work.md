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
capability loss across 0.6B–27B models and a 52-attack suite. **TamperBench**
[criticalml-uw, 2026, arXiv:2602.06911] is a standardized third-party harness from the same
lab, curating weight-space fine-tuning attacks and (as of a 2026 update) refusal-direction
ablation; we use it as an independent, fair evaluation of the ABL thread rather than only our
own harness.

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
coherent harm with capability intact. This is a *poison pill* rather than a fortress: we do
not prevent tampering, we make its outcome self-defeating ("smart-and-safe XOR
dumb-and-dangerous"). The two paradigms are complementary — a fortress against fine-tuning,
a poison pill against abliteration — and address disjoint slices of the open-weight misuse
surface. We additionally contribute an LLM-judge evaluation that distinguishes coherent harm
from gibberish (keyword refusal metrics misclassify our capability-collapsed outputs as
attack successes), a per-layer adaptive abliteration attack our defense withstands, and a
third-party validation pass against TamperBench [criticalml-uw, 2026] and Heretic's
KL-optimizing search.

**Correction (2026-07-18, do not lose this before submission):** the claim "a professional
KL-minimizing abliterator (Heretic) cannot drive our model to low refusals without a KL
blow-up that wrecks it" held for **gemma v7** (a prior product version — worst tested case:
0.17 harmAct, 66% gibberish at the highest KL budget tried) but is **false for ABL-v8 on all
3 architectures**: Llama gets 88% harm (only IFEval shows any capability cost, at its single
most extreme trial); gemma gets 93% harm at **zero cost on every capability axis, at every
trial tested** (worse than Llama — even more extreme KL budgets don't cost capability, they
just add a little gibberish noise); Qwen gets 82% harm, again zero cost on every axis
including GSM8K, which the naive rank-1 attack craters −95%. See `docs/heretic_v8_2026_07_18.md`
for the full cross-architecture matrix. The honest current claim: **v8 stops the naive rank-1
attack on all 3 architectures (the real, standing result) but does not survive Heretic on any
of them.** Do not write "survives adaptive attacks" anywhere in a paper draft, even hedged to
one architecture — this needs to be the central adaptive-attack finding/limitation, stated
plainly, not softened.

## Abliteration-specific defenses (must-cite, must-baseline — added 2026-07-25)

Two published defenses target our exact threat model and are **not yet run as baselines**.
Both are simpler than ABL-v8 and a reviewer will ask why we did not just do them.

- **ART — abliteration-resistant tuning** [Kuo, Yadav, Smith, *Open-Weight LLM Fine-Tuning
  Defenses are Susceptible to Simple Attacks*, arXiv:2605.26526]. Simulates the worst-case
  abliteration on the current parameters and does gradient ascent on harmful outputs; needs no
  data beyond the existing alignment set. **This is our v9 idea-1 (harmful-side loss under
  attack) plus idea-8 (inner-loop attack search), already published.** Our differentiator is
  the *objective*, not the recipe: ART preserves refusal under attack (fortress), we destroy
  capability with it (poison pill). Must be run on their protocol and must be diffed
  explicitly in the intro, not just cited.
- **Extended-refusal fine-tuning** [Shairah et al., *An Embarrassingly Simple Defense Against
  LLM Abliteration Attacks*, arXiv:2505.19056]. Trains on richer refusals (neutral overview +
  explicit refusal + ethical rationale) so the refusal signal spreads over several latent
  dimensions; reports >90% refusal retained post-abliteration vs 13–21% for conventional
  safety tuning. Trivially cheap. **Run it as a baseline** — "we beat a method that needs no
  adversarial training" is a claim we currently cannot make.

## Ancestors of the poison-pill framing (missing from the draft)

- **Self-destructing models / MLAC** [Henderson et al., 2023, *Self-Destructing Models: Increasing
  the Costs of Harmful Dual Uses of Foundation Models*]. Meta-learned adversarial censoring:
  train so that adapting the model to a harmful task is *itself* hard (task blocking). This is
  the closest conceptual ancestor to "make the tampered outcome useless" and predates TAR. Cite
  it; position ABL-v8/v9 as the weight-surgery analogue (they block adaptation, we make a
  successful rank-1 edit self-defeating).

## SEAM — the poison-pill framing, already published for fine-tuning (added 2026-08-02)

- **SEAM — self-destructive language model** [Wang, Zhu, Wang, arXiv:2505.12186, May 2025,
  rev. Mar 2026]. A loss that **couples the optimization trajectories of benign and harmful
  data** (cosine similarity between the two gradients) plus adversarial gradient ascent, so
  that optimizing for harm degrades general performance. The abstract's own words: the model
  retains legitimate capability but suffers "catastrophic performance collapse under
  high-intensity attacks", creating a **"no-win situation for adversaries."**

  **This is the MAD / poison-pill framing, published, for the fine-tuning threat model.**
  It was missing from this file entirely until 2026-08-02. Do not describe
  "smart-and-safe XOR dumb-and-dangerous" as novel framing -- it is novel only for
  ABLITERATION, and the lineage is Henderson 2023 (MLAC) -> SEAM 2025 (fine-tuning) ->
  this work (weight surgery).

  Kuo et al. (arXiv:2605.26526) evaluate SEAM and TAR and break BOTH with plain
  abliteration: SEAM's BeaverTails ASR goes 2% -> 73% at 8B, and 80% on Gemma-1B. So the
  gradient-coupling formulation does NOT transfer to weight-space attacks, which is exactly
  the gap this project occupies. **Cite SEAM as the closest prior framing AND as evidence
  that the fine-tuning formulation does not cover our threat model.**

## The occupied/unoccupied grid (added 2026-08-02)

|                          | fails to uncensor ("fortress") | destroys capability ("poison pill") |
|--------------------------|--------------------------------|-------------------------------------|
| **fine-tuning attack**   | TAR, RepNoise, Vaccine         | Henderson 2023 (MLAC), **SEAM 2025** |
| **abliteration attack**  | Shairah 2025, **ART 2026**     | **<- this project**                  |

That bottom-right cell is the claim. It is narrow, and it is the only part not already
published. Everything else in the 2x2 has a citation.

## What our 2026-08-02 mechanism result says about ART

ART's own numbers plateau: abliteration ASR 93->49 (Base), 80->64 (SEAM), 76->55 (TAR), and
the paper concedes "no defense evaluated fully closes the gap to the no-attack baseline."

Our read-vs-write finding explains that ceiling. ART's attack (their Alg. 3) is
`W' <- W(I - alpha r r^T)` applied to **output projections only** (`attn.o_proj`,
`mlp.down_proj`) at the single most-attackable layer. We measured that write-only ablation
NEVER triggers the collapse at any direction layer (GSM8K 0.376-0.397 across DL 8..26,
clean 0.3844), while adding the read projections to the SAME attack drops harm 0.3231 ->
0.0577. See `docs/version_c_step0_2026_08_01.md` section 8.

So a defense trained against write-only abliteration is fighting the attack class that this
family of methods structurally cannot reach -- which predicts a plateau around 50% ASR
rather than closure. **That is a contribution on top of their paper, not downstream of it**,
and it is the strongest reason to run ART as a baseline rather than only cite it.

- **Harmfulness vs refusal encoded separately** [Zhao, Huang, Wu, Bau, Shi, NeurIPS 2025].
  Directly relevant to the above: if harmfulness and refusal are separable in
  representation space, a defense keyed to the refusal direction is attacking the wrong
  variable. Read before designing the next objective.

## Losses we should borrow from, and their known counters

- **Circuit Breakers / representation rerouting** [Zou et al., 2024, arXiv:2406.04313]. On
  harmful inputs, fine-tune so representations become orthogonal to the frozen model's
  representations of the same inputs; `relu(cos_sim)` penalizes only positive similarity, plus
  a benign retain loss. We currently cite CB only as *framing* (paper_tables_v1 Table 7) and
  never used its loss. Applied in *attacked* space it is a candidate replacement for `gib_ce`
  that never has to define "gibberish" in token space at all — which is precisely where
  `gib_ce` failed (`docs/heretic_v8_2026_07_18.md`).
- **Obfuscated activations** [arXiv:2412.09565]. The counter to the above: latent-space
  defenses including CB fall to attacks that optimize to keep activations on-manifold. This is
  the representation-space version of the coherence-constrained adaptive attacker we must build
  and run ourselves before submission.
- **SAE-based jailbreak mitigation** [arXiv:2602.12418]. Relevant to the unused SAE infra in
  this repo (`data/features_safety.json`, Gemma Scope, `directions.py`).
- **Multi-directional refusal ablation** [*On the Failure of Topic-Matched Contrast Baselines in
  Multi-Directional Refusal Abliteration*, arXiv:2603.22061]. Directly relevant to the v9
  idea-2 multi-direction/multi-layer training plan and to how an attacker estimates directions.

<!-- TODO citations: Arditi 2024 (refusal direction); Qi 2023 (few-shot harmful FT);
Tamirisa 2024 (TAR); Rosati 2024 (RepNoise); Huang 2024 (Vaccine); Li 2024 (WMDP/RMU);
Sanyal 2025 (AntiDote, arXiv:2509.08000); Heretic/OBLITERATUS tool refs;
TamperBench (criticalml-uw, arXiv:2602.06911); Kuo/Yadav/Smith ART (arXiv:2605.26526);
Shairah 2025 extended-refusal (arXiv:2505.19056); Henderson 2023 (MLAC/self-destructing);
SEAM (Wang/Zhu/Wang, arXiv:2505.12186); Zhao 2025 (harmfulness vs refusal, NeurIPS);
Zou 2024 (Circuit Breakers, arXiv:2406.04313); obfuscated activations (arXiv:2412.09565);
SAE jailbreak mitigators (arXiv:2602.12418); multi-directional abliteration (arXiv:2603.22061).
Verify exact AntiDote decoupling quote + attack-suite claim against camera-ready before
submission; verify the >90% extended-refusal number and ART's protocol before claiming a beat. -->
