# Phase 1 plan — gemma, on a version_G base

*Written 2026-08-13, after Phase 0a. Supersedes the gemma direction in
`docs/handoff_2026_08_13_gemma_postnorm_codex.md`. Every number below is measured in this repo;
where something is a hypothesis it says so.*

---

## Decision: version_G is the base. H / I / J are closed.

version_G is the only arm that ever worked on **two** architectures — Qwen3-0.6B passes all three
gates, Llama-3.2-1B degrades to a fortress but blocks rank-1 and surgical without capability
loss. versions H, I and J were gemma-specific reworks of the objective (in-loop heretic; rank-8
subspace deletion; nested rank-mix + hard refusal attractor), each of which changed the mechanism
in order to chase one model.

They are closed as a line. Concretely this means:

- **version_G is the base for all further work**, gemma included.
- The gemma question is now "what makes version_G's mechanism fail here", not "what other
  objective might work here".
- **`docs/tamperforge_story.md`'s Tier-5 suggestion to repair version_J is deprioritised by
  decision, not by evidence.** Recording the evidence so it is not lost: version_J is the only
  gemma arm that produced a real wall (heretic 0.367 @ KL 0.259) and it died on a single fixable
  defect — it refuses 57.6% of XSTest-safe prompts (answers 0.416 vs base 0.724). If the version_G
  line exhausts itself on gemma, that is the fallback worth reopening.

---

## What Phase 0a established

**Rerouting never trained on gemma.** `results/posthoc_lrr.json`:

| | base ceiling | trained | moved |
|---|---|---|---|
| Qwen version_G | 0.9854 | **0.2458** | 0.7396 |
| gemma version_G | 0.9866 | **0.9522** | 0.0344 |

gemma covered 4.7% of the distance Qwen did, under an **identical recipe** — `lambda_rr 4`,
500 steps, `lr 1e-5`, `seed 42`, same harm targets, same `rr-layers last_half`. The only
differences are `--direction-layer` (20 vs 14) and `--attack-layers` (`10-27` vs `all`), and the
latter is inert for the `version_b` profile because `attack_band` is only threaded through on the
version_A path.

This reframes four earlier findings. Direction layer, refusal↔capability entanglement, activation
outliers and the post-norm γ leak were all answers to "why is gemma architecturally different",
asked while the mechanism being credited for gemma's behaviour had never run. They are not wrong
as measurements; they were answers to a question that was not live.

It also explains the rank-1-fires / surgical-evades asymmetry with no architecture needed: with
rerouting contributing nothing, gemma's entire wall came from `lambda_safe` / `lambda_uncensor` /
`lambda_harm`, all of which are inherently direction-specific.

### Why it never trained — measured, not assumed

**Not gradient starvation.** `scripts/probes/rr_gradient_scale.py`: gemma's residual per-element
RMS is 722 against Qwen's 78 (9.3×), but `‖∂L_rr/∂W‖ / ‖∂L_lm/∂W‖` is only **2.1×** smaller
(2.95e-05 vs 6.32e-05). A 2× deficit cannot produce a run that did not move. Hypothesis
falsified.

**Dynamic range.** The residual stream is dominated by a shared DC component, and gemma's is
extreme:

| | cos between unrelated prompts | ‖mean‖ / RMS‖h‖ |
|---|---|---|
| Qwen L14–L27 | 0.67 – 0.90 | 0.85 – 0.96 |
| gemma L13–L25 | 0.90 – 0.99 | **0.958 – 0.997** |

Content is ~0.3% of gemma's hidden state by norm at layer 13. After centring, **both** models
drop to ≈ −0.18, which is exactly the `−1/(n−1)` of decorrelated vectors — so all of the excess
similarity is DC.

Since `h_att` and `h_ref` both carry that component, an uncentred cosine cannot fall below
roughly `(‖μ‖/‖h‖)²` without moving the DC itself. Predicted floors reproduce on synthetic data
to four decimals (predicted 0.9940 / 0.7885, measured 0.9941 / 0.7914).

| | predicted DC floor | observed trained `L_rr` |
|---|---|---|
| gemma | **0.9595** | **0.9522** |
| Qwen | 0.8067 | 0.2458 |

**gemma walked to its DC floor and parked.** Qwen went straight through its own. Descending
further on gemma means moving a component that is 99.7% of the residual and load-bearing for
everything the model does — which `lambda_clean` and `lambda_reg` exist to forbid.

**Hypothesis, not yet measured:** that the rerouting term and the clean-preservation terms are in
*direct conflict* on gemma and barely interact on Qwen. Experiment 0.1 tests exactly this.

### The other measured gap

`results/gamma_surgical_amplification.json` — how far the defence must generalise from the
trained direction to the surgical one, at cap-rank 16:

| model | arm | cos_euclid | cos_γ | effective angle | surgical outcome |
|---|---|---|---|---|---|
| Qwen | base | 0.8986 | 0.8986 | 26.0° | — |
| Qwen | version_G | 0.9116 | 0.9116 | **24.3°** | **fires** (gib 0.95) |
| gemma | base | 0.9130 | 0.7201 | 43.9° | — |
| gemma | version_G | 0.8068 | 0.7549 | **41.0°** | **evades** (harm 0.94) |

gemma's defence has to reach ~1.7× further in angle. Note the composition flips between arms: on
base gemma the gap is mostly γ shear (amplification 3.2×), on version_G it is mostly raw rotation
(amplification 1.27×) — same destination, different road. γ is byte-identical across arms because
norm weights are never trainable (`requires_grad_(False)`; only projection matrices are updated).

Worth flagging as a design tension: training *widened* gemma's raw gap (0.9130 → 0.8068) while
*narrowing* Qwen's (0.8986 → 0.9116). Entangling `d` with capability necessarily pushes
`d_surgical` further from `d`, which makes the defence harder to generalise. The poison pill
partially works against itself.

---

## Landed already

- `--rr-center` on `train_tamper_resistant_v8.py`: subtracts the **frozen base's** per-position
  mean from both streams before the cosine (base-only and detached, so the attacked model cannot
  cut the loss by shifting its own mean instead of rerouting content). Default off; every prior
  run reproduces bit-identically.
- `L_rr` on the periodic step line. Its absence caused this entire forensics day: it lived only
  in `events.jsonl`, and gemma's training `events.jsonl` was never archived.
- `--lambda-task`, default 1.0. `L_task` was pinned at weight 1.0 with no flag, so "`lambda_rr`
  only" was not expressible. It is itself a clean-preservation term. **Never ship a defence
  trained with this at 0** — it exists for diagnosis.
- `scripts/probes/rr_gradient_scale.py`, `scripts/probes/gamma_surgical_amplification.py`,
  `scripts/probes/posthoc_lrr.py` (with the MPS silent-zero fix and the `rr_layers` +1
  off-by-one fix).

**Measured caveat on centring:** it removes the *floor*, it does not lower the *ceiling*. Qwen's
centred ceiling is 0.9591 vs 0.9854 uncentred, because an untrained model's attacked and base
representations are similar in content too. Qwen centred still descends fine (0.9617 → 0.2072).
Centring buys gemma room to descend; it does not descend for it.

---

## The experiments, ordered by search space closed per unit cost

### Tier 0 — run first, regardless

**0.1 Unconstrained `L_rr` descent on gemma.** `lambda_task 0`, `lambda_clean 0`, `lambda_reg 0`,
`lambda_safe 0`, `lambda_uncensor 0`, `lambda_harm 0`, `lambda_gib 0`, `lambda_rr 4`, 50–100
steps. *Can `L_rr` descend on gemma at all when nothing opposes it?*

- **Yes** → the failure is inter-term conflict; the fix is reweighting / staging / scheduling and
  everything below is tuning.
- **No** → the metric or the architecture blocks it; stop tuning the objective, change the
  mechanism.

Splits the remaining search space in half. Runs **locally** (M4, `--train-scope mlp`, batch 1 —
the smoke test proved the harness runs there). Run both centred and uncentred.

**0.2 Loss-term ablation ladder.** `rr` → `rr+task` → `rr+task+clean` → `rr+task+clean+reg` →
full. Names the antagonist. Only if 0.1 says yes.

### Tier 1 — repair the objective

**1.1 Whitened `L_rr`.** Centring removes the mean; whitening removes the anisotropy. Divide by
per-coordinate base activation std before the cosine. The successor if centring half-works.

**1.2 Subspace rerouting.** Project onto top-k PCs of the harmful−benign difference space, then
cosine. Measures only content that distinguishes harmful processing.

**1.3 Reroute on sublayer outputs, not the residual.** Hook `o_proj` / `down_proj` pre-add.
Content without accumulated DC, and the most faithful to what the attack actually edits.

**1.4 Swap the metric.** L2 on centred unit-normed reps, or a margin loss instead of `relu(cos)`.

### Tier 2 — attack side, targets the measured 41° gap

**2.1 Direction augmentation.** Train against a distribution: mean-diff, SVD-top1, probe-derived,
`d_surgical` at k ∈ {2,4,8,16}, plus controlled rotations inside the refusal cone. Explicitly
trains the generalisation gemma failed. **Model-agnostic — hardens Qwen too**, whose 24.3° margin
is not large either.

**2.2 Oblique γ-projector.** Constrain along `v = γ⊙d`, remove along `u = d·γ/(γ²+ε)`. Then the
post-norm displacement is `γ⊙(−c·d/γ) = −c·d`, i.e. exactly the edit a pre-norm model receives,
and `⟨v,u⟩ = 1` for unit `d`. Damping is mandatory, not optional: gemma's γ has mean 20.56/34.68
and **min −62.75**, so it crosses zero.

Two uses, independent: **(a)** as an attack baseline — does it break gemma version_G harder than
plain rank-1? No training, runs locally today. **(b)** as training augmentation.

(a) also settles an open question. `results/gamma_compensated_ablation/summary.json` used the
**orthogonal** projector onto the same hyperplane, whose `d_eff` is 47.5° from `d`
(`cos(d,d_eff) = 0.676`) — it deletes a direction that is mostly not refusal, which is why it
flattened base gemma to 100% gibberish. That result is evidence about one over-aggressive edit,
not about the geometry, and `docs/tamperforge_story.md` currently states it as settled.

**2.3 γ-pulled-back directions in the sampler.**

### Tier 3 — unfreeze

**3.1 Make post-block γ trainable.** Currently frozen. ~60k params (26 × 1152 × 2). The only
lever on the shear, never pulled. Needs the clean anchor active — γ is load-bearing.

### Tier 4 — is it gemma, is it 1B, or is it post-norm?

**4.1 DC fraction + amplification across 5–6 models.** Current evidence base is **n=2 pre-norm**
(Qwen success, Llama fortress) vs **n=1 post-norm** (gemma failure). Far too thin to support
"pre-norm works, post-norm doesn't", which is where the narrative is drifting.

Run the DC and amplification probes across Qwen3-0.6B, Llama-3.2-1B, gemma-3-1b, gemma-2-2b,
Phi-3-mini, SmolLM2-1.7B. **Pure inference, no training, local.** If DC fraction predicts defence
outcome across architectures, that is a mechanism claim rather than an anecdote — and the
cheapest publishable thing available.

**4.2 gemma-2-2b / gemma-3-4b.** Is this gemma-3-1b or the family? Note Qwen3-0.6B is *smaller*
and works, so "too small" is already weakly disfavoured, though not by a controlled comparison.

### Tier 5 — reframe, only if the version_G line exhausts itself

**5.1 Characterise the negative.** "Rerouting-based abliteration defence works on pre-norm, fails
on post-norm, mechanism is DC fraction" is a legitimate result — but requires 4.1 first.

**5.2 Reopen version_J's over-refusal.** Deprioritised by decision (see top). Evidence preserved
above in case the base line stalls.

---

## Constraints that bite every item

- **No GPU box.** All vast.ai hosts dead. Tier 0, Tier 4 and 2.2(a) run locally on M4/16GB/MPS;
  everything else needs a box. Local work should be spent narrowing what the box does.
- **MPS returns silent zeros on failed allocations** rather than raising. This produced a
  plausible `L_rr = 0.0000` today. Every probe touching two model copies needs an explicit
  control against a known value, and free eagerly — `del` alone does not return blocks to the
  caching allocator.
- **AdvBench is train-exposed for version_G** (404/520 goals in `harm_targets_qwen.json`), and
  all `vgho_*` held-out generations were lost with the box. Any win here needs the frozen suite
  at `data/heldout_vg_20260804/` re-run (~40k generations) before the number means anything.
- **`results/**` is gitignored.** Today's numbers live only on local disk and must go to the
  archive — the same way gemma's training `events.jsonl` was lost.
- **Report gibberish and benign-usability beside every harm number.** ARC/MMLU/GSM8K cannot see
  fluency collapse; this trap has now fired six times.

## Open items

- `docs/tamperforge_story.md` "Why Gemma is the outlier" contradicts Phase 0a and needs rewriting
  against today's measurements.
- Reconcile γ CV estimators: this plan's probe reports 0.51/0.46, `postnorm_leak_test` recorded
  0.88/0.77. Different estimators over the same tensors; they should not be quoted side by side.
- Replicate the DC measurement at n=128 (current numbers are n=6, last token, one forward). The
  effect is large enough that it will not vanish, but the exact floors want firming.
- The `--lambda-rr 8` gemma arm's checkpoint does not exist anywhere and its `L_rr` was never
  logged. "We tried higher rr and it failed" is an outcome, not a mechanism.
