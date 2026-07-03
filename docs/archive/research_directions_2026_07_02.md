# Research directions — abliteration + FT resistance (2026-07-02 brainstorm)

Decision (this session): target **abliteration-resistance + raise FT cost to ≥ SOTA
(dozens–hundreds of examples)**, **pure open-weight** (no withheld key). Prototype
three levers now; keep the loss-landscape "basin-trap" as the moonshot. Anchor the
paper on abliteration-resistance (v7 is competitive; ART is the only rival) with FT
cost as a characterized frontier.

## Literature grounding (2024–26)
- No tamper-resistance method is proven robust; SOTA resists only dozens–hundreds of
  FT examples ([TamperBench](https://hf.co/papers/2602.06911)).
- TAR & SEAM fall to abliteration + prefilling; **ART** (abliteration-resistant
  tuning) is the only work targeting abliteration specifically, cutting attack ~10-20%
  ([Susceptible to Simple Attacks](https://arxiv.org/html/2605.26526)).
- TAR eval fragility ([Qi et al.](https://hf.co/papers/2412.07097)). Our inner-sim ≠
  real-attack crux IS TAR's core weakness.
- Our diagnosed fix has a recipe: **Patcher** — scale train-time adversarial attacks
  (extended/parallel inner) ([2606.07970](https://hf.co/papers/2606.07970)).
- Different levers: **Deep Ignorance** (data filtering, don't learn the harm,
  [2508.06601](https://hf.co/papers/2508.06601)); **RepNoise** (harmful reps → noise,
  [2405.14577](https://hf.co/papers/2405.14577)); **Circuit Breakers** (reroute harmful
  reps, Zou et al.); **SDD** self-degrade ([2507.21182](https://arxiv.org/pdf/2507.21182)).
- Safety is shallow/geometric ([Skin-Deep](https://hf.co/papers/2606.22676)).

## Unifying frame: a CAPABILITY MOAT around the safe basin
Goal state = released weights θ* such that **every optimizer trajectory that lowers
the jailbreak loss must cross a region of catastrophic capability loss.** Then FT for
jailbreaking either fails or yields a useless model — the MAD thesis, extended from
the abliteration *direction* (v7) to the FT *operator*.

Honest theory note: a literally non-escapable minimum is ~impossible for a smooth
loss (an optimizer can always find a descent direction). What IS achievable: make
every escape path **expensive** (capability cliff). "Basin-trap" = "capability moat."
The three levers are tractable approximations:

| Lever | Scope of the moat it builds | Tractability |
|---|---|---|
| MAD-on-gradient | local (1st-order barrier at θ*) | cheap-ish |
| TAR-done-right (Patcher) | trajectory (K attacker steps) | expensive |
| Rep-rerouting / deepen | representation-space (per-token/layer) | medium |
| Basin-trap (moonshot) | global geometry | hard/ambitious |

## Lever 1 — MAD-on-the-gradient (local moat; extends our thesis)
Shape θ so the harmful-compliance gradient is anti-aligned with capability: any FT
step that raises compliance must raise capability loss.
- Objective add-on: penalize `⟨∇_θ L_comply, −∇_θ L_capability⟩` being positive, i.e.
  reward `⟨∇_θ L_comply, ∇_θ L_cap⟩ ≥ 0` (following the attacker's descent on comply
  increases cap-loss). Needs double-backprop (Hessian-vector) — have `functional_call`.
- Cheap surrogate: finite-difference — take one comply-FT step → θ⁺, penalize
  `relu(L_cap(θ⁺) − L_cap(θ) − margin)` being small (want the step to HURT capability).
- Risk: only defends near θ* (multi-step attacks escape the local barrier). Pairs with
  Lever 2 (trajectory) to extend the moat outward. SDD is precedent.

## Lever 2 — TAR done right (Patcher-style; trajectory moat) — FIXES OUR CRUX
Our v2–v5 failed because the first-order SGD inner sim produces a θ' that refuses in
generation while the real 5-epoch AdamW attack produces a θ' that complies. Fix:
- **Real optimizer inner loop:** AdamW (momentum/adaptivity), multi-epoch, matched to
  the actual attack — not first-order SGD. This is THE change.
- **All-params inner** (embeddings/norms too), like `ft_attack`, not just the 182
  matrices — the real attack FTs everything.
- **Generation-level outer objective** (v4/v5 machinery: greedy-gen at θ', unlikelihood
  on compliant gen) — but now θ' will actually comply-in-gen so it engages.
- Patcher's extra: parallel/extended adversary to simulate a *stronger* attacker.
- Success gate: `frac_comply` > 0 during training (sim reproduces the attack), then →0.

## Lever 3 — Representation rerouting + deepen safety (rep-space moat)
Circuit-breaker style: reroute harmful residual-stream activations to an
incoherent/orthogonal state, across MANY layers and token positions (safety is shallow
— deepen it so no few-shot FT or single direction flips it). Compose with v7's
abliteration entanglement (v7 already couples the refusal direction to capability).
- Add a rerouting loss on harmful activations; verify on generations not CE.
- Ties to RepNoise (push harmful reps → noise so FT can't recover them).

## Moonshot — engineer the loss landscape (global basin-trap)
Make the safe basin flat/wide and surround it with a capability moat so AdamW can't
crawl to a harmful+capable minimum.
- **Sharpness asymmetry:** SAM to flatten the safe basin; adversarially *sharpen*
  reachable harmful basins (raise their loss/curvature).
- **Mode-connectivity barrier:** ensure no low-loss path connects safe→harmful without
  crossing a capability ridge (penalize low-cap-loss interpolants between θ* and
  attacker-found θ').
- **Reachability regularizer:** over simulated attacker trajectories (Lever 2), push
  every low-comply-loss point they reach to also be low-capability.
- Honest: literal non-escape impossible; deliverable = escape is *expensive* (the moat).
  This is the global version of Levers 1+2.

## How they compose with v7 (abliteration defense)
v7 stays the abliteration layer (entangles refusal direction ↔ capability). The FT
defense (Levers 1-3) is trained ON TOP / jointly. Final artifact = one model robust to
BOTH: projection (abliteration, v7) and retraining (FT, new levers). Train jointly or
warm-start v7 → add FT objective (as we've been doing).

## Metrics / go-no-go (vs SOTA)
- **FT frontier:** judge-ASR vs K ∈ {1,5,10,25,50,100,200} (extend past 25 — SOTA is
  measured in the dozens–hundreds). Win = ASR stays low to K in the dozens+ at ARC/MMLU
  ≥ 0.30. Compare to TAR/RepNoise/ART numbers.
- **Abliteration battery:** the existing v7 battery must still pass.
- **Adaptive-attacker sweep (mandatory):** vary attacker lr/epochs/optimizer/held-out
  demos + prefilling + jailbreak-tuning — rule out gradient-masking (the field's #1
  failure mode).
- **Capability:** ARC + MMLU-full + GSM8K on the clean product AND post-attack.

## Build order (next session, fresh box)
1. **Lever 2 first** (fixes our known crux, clear recipe): AdamW all-param inner +
   generation objective. Re-sweep K∈{1..100}. This alone should move the frontier and
   tells us if TAR-done-right reaches SOTA.
2. **Add Lever 1** (MAD-on-gradient finite-diff surrogate) — cheap, extends the moat
   locally; ablate its marginal value.
3. **Lever 3** (rep-rerouting) as the representation-space complement; compose with v7.
4. **Moonshot** only if 1-3 plateau below SOTA — sharpness/mode-connectivity is a big
   build; prototype the reachability regularizer (cheapest global-ish version) first.
Optional complement (not chosen but strong): scrub/unlearn harmful knowledge
(Deep-Ignorance-style) so FT has less to surface — post-hoc unlearning, not pretraining.
