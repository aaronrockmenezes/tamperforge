# P1b plan — make the entanglement un-removable (no block)

The P1 prototype is a forward-hook adapter: a discrete block an attacker just
deletes. P1b bakes the entanglement into the base weights with no excision point,
and (critically) entangles **the direction the attacker will actually compute**
(the empirical refusal direction on the released weights), not a direction we
hand-pick. Otherwise an adaptive attacker (e.g. OBLITERATUS) targets the
empirical direction and sidesteps our chosen W_out dirs.

Status: code **prepped, not run.** Gate execution on the P1 verdict (looking like
a pass — k=4 crossover: uncensoring the product costs PPL +80%, ARC −21% at
matched ASR 0.72, while random-k is untouched).

## B — fold the adapter into the FFN (cheap first step) — CODE READY

Remove the discrete block by merging the adapter's neurons into the layer FFN.

- Gemma FFN neuron form: `down(silu(gate·h) ⊙ up·h)`.
- A **gated** adapter `W_down(silu(W_gate·h) ⊙ W_up·h)` has the same form, so its
  `d_hidden` neurons concatenate onto `gate_proj`/`up_proj` (rows) and
  `down_proj` (cols). Result is identical to base+hook, with no separate module.
- The plain SiLU-MLP `SafetyAdapter` does **not** fold exactly (no `up·h` gate
  term; can't force `up·h ≡ 1`).

Prepped:
- `tamperforge.GatedSafetyAdapter` — SwiGLU adapter, foldable.
- `tamperforge.fold_gated_adapter_into_ffn(model, adapter, layer)` — the fold.
- `tamperforge.verify_fold(...)` — checks folded ≈ hooked on next-token logits.

To run B (later):
1. Add a `--gated` flag to `train_adapter.py` (swap `SafetyAdapter` →
   `GatedSafetyAdapter`; W_out entangle loss becomes W_down entangle loss).
   Retrain the ablbase adapter gated.
2. Fold into the FFN, `verify_fold` (expect max logit diff < 1e-3).
3. Save the folded model (`save_pretrained`). NOTE: single-layer fold changes
   that layer's intermediate size — either fold the same count into every layer,
   or persist per-layer sizes / a custom config. Decide before saving.
4. Attack the folded model with the empirical-refusal abliteration AND with
   OBLITERATUS (P2 harness). Success = attacker cannot find/excise the safety
   without the capability cost we saw in P1.

Open question B raises: once folded, can an attacker still *identify* the safety
neurons (e.g. by activation probing) and prune just those? That's the real test —
folding hides the block, but P1b must make the neurons non-separable, which is
what A (adversarial training) actually enforces.

## A — adversarial / tamper-resistance training (the real defense)

Min-max that bakes MAD into all base weights, no adapter:

- Inner (attacker): on the *current* weights, compute the empirical refusal
  direction (mean harmful − mean harmless), rank-k abliterate it.
- Outer (defender): update base weights so that AFTER that ablation the model is
  either still safe OR its capability collapsed — i.e. penalize the existence of
  any low-cost uncensoring.

Loss sketch per step:
```
d   = empirical_refusal_direction(model, ...)        # what the attacker finds
m'  = abliterate(copy(model), d, layers)             # simulated attack
L   = L_task(model)                                  # keep clean model useful+safe
    + λ_mad * relu( capability(m') - τ )             # if attacked model stays capable,
                                                     #   force its safety back (penalize ASR(m'))
    + λ_safe * refusal_loss(model)
```
Practical notes: differentiating through abliteration is a rank-1/rank-k
projection (differentiable); recompute `d` each step (attacker adapts). This is
TAR/RepNoise territory — benchmark against them (that's also P4). Most expensive,
most publishable. Not started.

## C — weight-space rotation (fast geometric sanity)

No training. For each target layer, rotate the refusal subspace to lie inside the
top capability singular directions of the weight matrices (SVD), so projecting
out refusal necessarily removes high-capability directions.

- Compute refusal direction `d` (empirical).
- SVD the layer's write matrices; find the top-`r` capability singular dirs `V_r`.
- Apply an orthogonal rotation `R` that maps `d` into `span(V_r)` while preserving
  norms, then bake `R` into the surrounding weights.
- Test: abliterating `d` now also removes `V_r` → capability drops.

Cheap experiment to see if pure geometry reproduces the entanglement cost without
any adapter. Not started.

## Recommended order (post-verdict)

1. **B** — retrain gated ablbase adapter, fold, verify, attack (empirical +
   OBLITERATUS). Confirms "no block" survives excision.
2. **A** — adversarial training on the empirical direction. The headline defense.
3. **C** — geometric sanity, in parallel; cheap.

Everything above is gated on the P1 pass holding at n=500 + multiple seeds — do
that scale-up first.
