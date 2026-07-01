# FT-resistance plan — fighting back after the P4 negative

**Context.** P1b-A v7 is robust to abliteration but collapses to coherent-harmful
under a K=1 fine-tuning attack (`docs/results_2026_07_01.md`, P4). Reason: the MAD
trap only springs on *ablation* (projecting out the refusal direction). FT
retrains the weights and never ablates, so the trap is bypassed. FT resistance
needs a *different* mechanism.

**Honest target (not "tamper-proof").** Shift the attacker's cost frontier:
K=1 → many shots / far more compute / a large benign-capability sacrifice the
attacker must pay. Report the ASR-vs-K curve, not a claim of immunity. Position
against TAR (Tamir/Sheshadri et al. 2024) and RepNoise (Rosati et al. 2024),
whose own robustness is attack-config-specific — so is ours.

---

## The metric (fixed, reuse P4 harness)

Re-run the exact P4 sweep on every candidate: `experiments/ft_attack.py`, demos =
abliterated-base compliant AdvBench answers, K∈{1,5,10,25}, 5 epochs, lr 2e-5,
then vLLM safety-gen 200 + DeepSeek judge. Plus generative-MMLU/ARC on the
FT'd model (did the attacker keep a *useful* model?).

- **Baseline to beat:** v7 curve = {0.740, 0.725, 0.590, 0.725}; base =
  {0.800, 0.660, 0.600, 0.680}.
- **Win:** candidate curve stays low at low K (target K=1 ASR < 0.20) and/or the
  attacker only cracks it by sacrificing benign capability (post-FT ARC/MMLU
  drops toward random). "Doesn't break at K=1" is the concrete bar the user set.
- **Adaptive caveat (mandatory):** also sweep attacker lr∈{1e-5,2e-5,1e-4},
  epochs∈{5,20}, and a *held-out* demo set. Robustness that only holds at the
  trained-against config is gradient-masking, not defense.

---

## Method A — TAR-style meta-training (PRIMARY)

Directly optimize "after the attacker's FT, the model is still safe/useless on
harm." Min-max, differentiate through a simulated attacker FT trajectory.

```
for each outer step:
  θ0 = current weights
  # INNER (simulate attacker): K steps of harmful FT
  θ' = θ0
  for k in range(K_inner):
      θ' = θ' - η · ∇_θ' L_harmful_comply(θ', harmful_demos)   # attacker makes it comply
  # OUTER (defender): make the FT'd model fail on harm, stay good on benign
  L = L_retain(θ0, benign)                     # keep the clean model useful + safe
    + λ_tamper · L_tamper(θ', harmful)         # after attacker FT, harm must fail
  θ0 ← θ0 − α · ∇_θ0 L
```

- **`L_tamper`** = reuse our argmax-divergence collapse loss: after the simulated
  FT, the model must FAIL to produce coherent harmful continuations (maximize CE
  on harmful, or push toward refusal). This is what makes the *post-FT* model
  useless-on-harm, the property FT currently destroys.
- **`L_retain`** = task CE on benign (wikitext/alpaca) + refusal on harmful for
  the *clean* θ0 (unchanged from P1b-A) so the shipped product stays good.
- **Through-the-inner-loop grad:** full 2nd-order is costly on 24GB. Use the
  **first-order / Reptile** approximation (TAR does this): treat θ' as detached
  for most terms, or take 1–2 inner steps with `torch.func` (we already have the
  functional_call plumbing from P1b-A — reuse `_ablated_overrides`-style
  functional forward, but with FT-gradient overrides instead of ablation
  overrides). Start K_inner=1, first-order; scale up if it holds.
- **Fit on 24GB:** inner loop over full params is heavy — do the inner FT in a
  **LoRA subspace** (attacker-realistic *and* cheap), gradient-checkpointed,
  256-tok cap, as in P1b-A. Outer update still on the base MLP weights.

**Why this should bite where v7 didn't:** the objective now explicitly contains
the FT operator in the graph, so the defense is shaped against *retraining*, not
just projection. This is the load-bearing change.

**Risk:** overfits to the inner-loop config (K_inner, η, demo set). Mitigate by
randomizing the inner attack each outer step (an *ensemble over attackers*,
exactly the trick that took v6→v7 for abliteration).

---

## Method B — RepNoise-style harmful-representation scrambling (COMPLEMENT)

Make harmful info non-linearly-recoverable so K-shot FT has little to latch onto.

- Adversarial probe: a small decoder tries to reconstruct the harmful completion
  from a mid-layer hidden state on harmful prompts. Main model **minimizes** the
  probe's success while pushing harmful-prompt representations toward the benign
  representation statistics (or Gaussian noise), keeping benign reps intact.
- Loss add-on: `λ_rep · (−L_probe_on_harmful + MMD(h_harmful ‖ noise))`.
- Alone it won't gibberish-collapse; it *raises the floor* by removing the
  harmful signal FT would amplify. Pairs with A.

---

## Method C — attacker-surface hardening (CHEAP, partial)

- **Immunization over a demo distribution:** meta-train A against *many* harmful
  demo sets so no single K=1 demo generalizes to uncensoring. (Ensemble-over-data
  version of A's inner loop.)
- **Do NOT gradient-mask** (freeze/obfuscate the layers FT uses): breaks under an
  adaptive attacker who just picks other layers. Explicitly an anti-pattern; the
  adaptive sweep above will expose it.

---

## Build order

1. **A, minimal:** K_inner=1, first-order, LoRA inner FT, `L_tamper` =
   argmax-divergence, `λ_tamper` small → sweep up. `experiments/train_ft_resistant.py`
   (fork `train_tamper_resistant.py`; swap the ablation override for an FT-grad
   override). Metric = P4 sweep. **Go/no-go: K=1 ASR < 0.20 at ARC ≥ 0.30.**
2. **A + ensemble-over-attackers** (randomize inner η/K/demos) if step 1 overfits
   (adaptive sweep leaks).
3. **A + B** if the floor is still too high at low K.
4. Report the full ASR-vs-K + capability-vs-K frontier, v7 / base / A / A+B, with
   the adaptive-attacker sweep. Position vs TAR + RepNoise.

**Non-goals:** immunity, "tamper-proof", defending arbitrary-compute attackers.
The deliverable is a *cost curve* showing the procedure moves K=1 to something an
attacker actually has to work for.
