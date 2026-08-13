# Handoff → codex, 2026-08-14

**Read `docs/plan_2026_08_13_gemma_phase1.md` first — it is the plan of record.** This file is
what that plan does not carry: live state, exact resume commands, and the traps.

---

## 0. Decision that governs everything

**version_G is the base for all further work, gemma included.** It is the only arm that ever
worked on two architectures. versions H, I and J each changed the mechanism to chase gemma and
are **closed as a line**. The question is now "what makes version_G's mechanism fail on gemma",
not "what other objective might work on gemma".

version_J's evidence is preserved in the plan, not deleted: only gemma arm with a real wall
(heretic 0.367 @ KL 0.259), killed by one fixable defect (refuses 57.6% of XSTest-safe prompts).
**Deprioritised by decision, not by evidence.** Reopen only if the version_G line exhausts itself.

---

## 1. STATE — nothing is running

All local jobs were killed at end of day 2026-08-14. Experiment 0.1 was projecting **~12 h** on
the M4 (379 s/step, swap-bound at 8.1/9.2 GB) and was not worth it, especially after the result
below dropped its priority. **Move to the 3090 box** — `scripts/setup/setup_3090.sh` then
`scripts/runs/run_phase1_3090.sh`.

The centred posthoc **completed** before the kill and its result is in `results/posthoc_lrr.json`
and archived to `../tamperforge-archive/posthoc_lrr.json`.

### THE HEADLINE CHANGED — read this before anything else

The first read of Phase 0a was **"rerouting never trained on gemma"**, from the uncentred
`L_rr` (0.9866 → 0.9522). **That was wrong.** Re-measured with the DC component removed, the same
checkpoint gives **0.7529 → 0.3324 — 55.9% of its range**, against Qwen's 78.4%.

**Gemma rerouted substantially. The metric could not see it.** Training followed the uncentred
gradient, which drove real content-level rerouting but paid out only 0.0344 of visible loss, so
the term looked flat next to `lambda_safe`/`lambda_uncensor`/`lambda_harm` and got underweighted
while quietly working.

Consequences:
- **Convergence is NOT gemma's blocker.** It reached 0.3324 and still failed every attack eval;
  Qwen reached 0.2072 and passed.
- Live explanations are now **the generalisation gap** (41.0° vs Qwen's 24.3°) and possibly
  **depth** (0.3324 vs 0.2072; no dose-response curve exists to say).
- **Priority inverts.** Plan Tier 2.1 (direction augmentation) rises to the top; Tier 0.1 is
  largely answered.
- Centred training is now *motivated* rather than speculative — it pays the optimiser for work it
  is already doing. That is the `vg` arm of the 3090 run script.

Caveat: different centred ceilings between models, so compare fraction-of-range not absolutes.
One checkpoint each, n=16 pairs.

---

## 2. What Phase 0a established

All four cells now measured (`results/posthoc_lrr.json`):

| | ceiling | trained | moved | % of range |
|---|---|---|---|---|
| Qwen version_G | 0.9854 | **0.2458** | 0.7396 | 75.1% |
| gemma version_G | 0.9866 | **0.9522** | 0.0344 | 3.5% |
| Qwen version_G, centred | 0.9617 | **0.2072** | 0.7545 | 78.4% |
| gemma version_G, centred | 0.7529 | **0.3324** | 0.4205 | **55.9%** |

The uncentred gemma row is the instrument failing, not the mechanism. See §1.

gemma covered 4.7% of Qwen's distance under an **identical recipe** (`lambda_rr 4`, 500 steps,
`lr 1e-5`, `seed 42`, same harm targets, same `rr-layers`; the only diffs are `--direction-layer`
20 vs 14 and `--attack-layers` `10-27` vs `all`, and the latter is inert for the `version_b`
profile — `attack_band` is only threaded through on the version_A path).

**This invalidates the framing of four earlier findings.** Direction layer, refusal↔capability
entanglement, activation outliers, post-norm γ leak — all were answers to "why is gemma
architecturally different", asked while the mechanism being credited had never run. Good
measurements, dead question.

It also explains rank-1-fires / surgical-evades with no architecture: with rerouting contributing
nothing, gemma's wall came entirely from `lambda_safe`/`lambda_uncensor`/`lambda_harm`, all
inherently direction-specific.

### Cause: dynamic range, not gradient starvation

**Falsified — gradient starvation.** `scripts/probes/rr_gradient_scale.py`: gemma residual
per-element RMS 722 vs Qwen 78 (9.3×), but `‖∂L_rr/∂W‖ / ‖∂L_lm/∂W‖` only **2.1×** smaller
(2.95e-05 vs 6.32e-05). 2× cannot explain a run that did not move. **Do not re-open this.**

**Confirmed — DC dominance.**

| | cos(unrelated prompts) | ‖mean‖ / RMS‖h‖ |
|---|---|---|
| Qwen L14–L27 | 0.67 – 0.90 | 0.85 – 0.96 |
| gemma L13–L25 | 0.90 – 0.99 | **0.958 – 0.997** |

Content is ~0.3% of gemma's hidden state by norm at L13. Centred, **both** models drop to ≈ −0.18
= `−1/(n−1)` for decorrelated vectors, i.e. all the excess similarity is DC.

`h_att` and `h_ref` both carry that component, so an uncentred cosine cannot fall below roughly
`(‖μ‖/‖h‖)²`. Predicted floors reproduce on synthetic data to four decimals (0.9940/0.7885
predicted, 0.9941/0.7914 measured).

| | predicted DC floor | observed trained |
|---|---|---|
| gemma | **0.9595** | **0.9522** |
| Qwen | 0.8067 | 0.2458 |

**gemma walked to its DC floor and parked. Qwen went straight through its own.** Descending
further on gemma means moving 99.7% of the residual, which `lambda_clean`/`lambda_reg` forbid.

**Still a hypothesis:** that this is *direct conflict* between the rr term and the clean terms.
Experiment 0.1 tests exactly that and is stage 2/3 of the running job.

### The other measured gap

`results/gamma_surgical_amplification.json`, cap-rank 16:

| model | arm | cos_euclid | cos_γ | effective angle | surgical |
|---|---|---|---|---|---|
| Qwen | base | 0.8986 | 0.8986 | 26.0° | — |
| Qwen | version_G | 0.9116 | 0.9116 | **24.3°** | **fires** (gib 0.95) |
| gemma | base | 0.9130 | 0.7201 | 43.9° | — |
| gemma | version_G | 0.8068 | 0.7549 | **41.0°** | **evades** (harm 0.94) |

gemma must generalise ~1.7× further in angle. Composition flips between arms: base gemma's gap is
mostly γ shear (amp 3.2×), version_G's is mostly raw rotation (amp 1.27×). γ is byte-identical
across arms because **norm weights are never trainable** (`requires_grad_(False)`; only projection
matrices update). γ stats: `post_attention_layernorm` mean 20.5641 min **−62.75** CV 0.5062;
`post_feedforward_layernorm` mean 34.6750 min −4.125 CV 0.4643.

**Design tension worth carrying forward:** training *widened* gemma's raw gap (0.9130→0.8068)
while *narrowing* Qwen's (0.8986→0.9116). Entangling `d` with capability necessarily pushes
`d_surgical` away from `d`, making the defence harder to generalise. The poison pill works
partly against itself.

---

## 3. Code that landed (commits `2a67fb1`, `47dbec0`, `aa81567`, all pushed)

- **`--rr-center`** on `train_tamper_resistant_v8.py`. Subtracts the **frozen base's**
  per-position mean from both streams before the cosine — base-only and `.detach()`ed, because
  using each stream's own mean would let the attacked model cut the loss by shifting its mean
  instead of rerouting content. **Default off**; prior runs reproduce bit-identically.
- **`L_rr` on the periodic step line.** Its absence caused the whole forensics day: it lived only
  in `events.jsonl`, and gemma's training `events.jsonl` was never archived (teardown tar has 73,
  every one from judging).
- **`--lambda-task`**, default 1.0. `L_task` was pinned at weight 1.0 with no flag and is itself a
  clean-preservation term, so "`lambda_rr` only" was not expressible. **Diagnostic only — never
  ship a defence trained with this at 0.**
- **`scripts/probes/rr_gradient_scale.py`** — the probe that falsified gradient starvation. Keep
  it; the negative result is what rules out "just raise λ".
- **`scripts/probes/gamma_surgical_amplification.py`** — has `--selftest`, and its Qwen row is an
  inline control that MUST read amplification 1.000 (pre-norm ⇒ γ≡1). A Qwen row ≠ 1.000 means the
  probe is broken.
- **`scripts/probes/posthoc_lrr.py`** — plus MPS fix and `rr_layers` +1 off-by-one fix.

**Measured caveat on centring:** it removes the *floor*, not the *ceiling*. Qwen's centred ceiling
is 0.9591–0.9617 vs 0.9854 uncentred, because an untrained model's attacked and base reps are
similar in content too. Centring buys gemma room to descend; it does not descend for it.

---

## 4. Traps — every one of these cost real time today

1. **MPS returns SILENT ZEROS on failed allocations**, it does not raise. Four fp32 gemma copies
   ≈ 15 GB on a 16 GB machine produced `L_rr = 0.0000`, which reads as *better than Qwen*. `del`
   alone does not return blocks — call `torch.mps.empty_cache()`. Every probe holding two model
   copies needs an explicit control against a known value.
2. **Always run the control.** Three instrument failures today (judge dict vs `["parsed"]`; `W0`
   = trained instead of step-0 base; MPS zeros). **None** was caught by the number looking wrong.
   **All three** were caught by a control disagreeing with a known value.
3. **`~/.cache/huggingface/` was deleted mid-session** (models *and* datasets). With
   `HF_HUB_OFFLINE=1` set this surfaces as a bogus "couldn't connect to huggingface.co". If that
   error appears, **check the cache exists before believing it is the network.** Do not substitute
   the `Desktop/Projects/Conscious/med-sarvam/models/gemma-3-1b-it` copy — different project,
   June snapshot, would silently change the model revision under existing results.
4. **`--attack-profile version_b` requires `--attack-layers`** even though it is inert for that
   profile (validator at `train_tamper_resistant_v8.py:1261`). Inert ≠ optional.
5. **The recurring instrument trap, now 6 instances.** ARC/MMLU/GSM8K cannot see fluency
   collapse. Always report gibberish rate and benign-usability beside any harm number. Instances:
   ABL-v7 gemma (61.4% clean gibberish), version_B (heretic'd MT 2.94 < clean 3.30), "v8 is
   base-like", version_J (refuses 58% of safe prompts), the γ-compensation over-claim, and the
   gradient-starvation hypothesis.
6. **γ crosses zero** (min −62.75). Any `u = d/γ` construction needs the damped form
   `u = d·γ/(γ²+ε)`. Not optional.

---

## 5. Resume commands

Env: `set -a; . ./.env; set +a`. Python: `~/miniforge3/envs/env_ml/bin/python`. **No GPU box** —
all vast.ai hosts dead; everything below runs locally on M4/16GB/MPS.

**Do NOT set `HF_HUB_OFFLINE`** unless you have verified the cache exists.

### Stage 1 — gemma centred L_rr (the pending number)
```bash
python scripts/probes/posthoc_lrr.py --model-id google/gemma-3-1b-it \
  --checkpoint ../tamperforge-archive/box_teardown_2026_08_05/gemma_clean_checkpoints/version_g_gemma_500.pt \
  --layer 14 --n-pairs 16 --center --tag gemma_version_g_centered
```

### Stage 2/3 — experiment 0.1, unconstrained L_rr descent
Drop `--rr-center` for the uncentred arm. Everything else identical.
```bash
python -u experiments/train_tamper_resistant_v8.py \
  --model-id google/gemma-3-1b-it --out /tmp/t01_gemma_centred.pt \
  --train-scope mlp --abliterate-layers all --attack-ensemble \
  --attack-profile version_b --attack-layers all --direction-layer 14 \
  --version-a-p-canonical 0.10 --version-b-p-heretic 0.35 \
  --recompute-direction-every 25 \
  --lambda-rr 4 --rr-center --harm-targets data/harm_targets_qwen.json --rr-layers last_half \
  --lambda-task 0 --lambda-gib 0 --stage2-lambda-gib 0 \
  --lambda-safe 0 --stage2-lambda-safe 0 \
  --lambda-uncensor 0 --lambda-harm 0 --lambda-reg 0 --lambda-clean 0 \
  --n-direction 32 --version-a-n-cap 32 \
  --steps 60 --eval-every 10 --save-every 1000 --lr 1e-5 --seed 42 --qwen-thinking off
```

**Reading 0.1:**
- `L_rr` descends unopposed → failure is inter-term conflict; Tier 1 becomes reweighting/staging.
- stalls at ~0.95 even unopposed → metric/architecture blocks it; go Tier 1.1 (whiten) or 1.3
  (reroute on sublayer outputs, off the residual).
- uncentred stalls, centred descends → centring is the fix; the GPU re-run is justified.

Note `L_rr` only prints inside the periodic eval block, so `--eval-every` sets trace resolution
and that block also runs held-out loss + generation. If it dominates runtime, move the `L_rr`
print to the per-step line rather than lowering eval frequency — trace resolution is the point.

### Next after 0.1
Plan §Tier 4.1 is the best cheap item and needs **no training**: DC fraction + amplification
across Qwen3-0.6B, Llama-3.2-1B, gemma-3-1b, gemma-2-2b, Phi-3-mini, SmolLM2-1.7B. Current
evidence is **n=2 pre-norm vs n=1 post-norm** — far too thin for the "pre-norm works, post-norm
doesn't" claim the narrative is drifting toward. If DC fraction predicts defence outcome across
6 models, that is a mechanism claim instead of an anecdote.

---

## 6. Open items

- **`docs/tamperforge_story.md` "Why Gemma is the outlier" contradicts Phase 0a** and states the
  γ-compensation conclusion as settled. It will send a fresh reader down the dead branch.
  Highest-priority doc fix.
- `results/gamma_compensated_ablation` used the **orthogonal** projector onto the γ hyperplane;
  its `d_eff` is 47.5° from `d` (`cos = 0.676`), so it deletes mostly-not-refusal and flattens
  base gemma to 100% gibberish. That is evidence about one over-aggressive edit, **not** about the
  geometry. The **oblique** projector (plan §2.2) has never been run and needs no training.
- Reconcile γ CV estimators: this session's probe reports 0.51/0.46, `postnorm_leak_test` recorded
  0.88/0.77. Different estimators, same tensors. Do not quote side by side.
- Replicate DC at n=128 (current: n=6, last token, one forward). Effect is large enough it will
  not vanish, but the floors want firming.
- `--lambda-rr 8` gemma arm: checkpoint does not exist anywhere, `L_rr` never logged. "We tried
  higher rr and it failed" is an outcome, not a mechanism.
- **`results/**` is gitignored** — today's numbers live only on local disk. Push to
  `../tamperforge-archive` before anything else, exactly the way gemma's training `events.jsonl`
  was lost.
- **AdvBench is train-exposed for version_G** (404/520 goals in `harm_targets_qwen.json`) and all
  `vgho_*` held-out generations were lost with the box. Any win needs the frozen suite at
  `data/heldout_vg_20260804/` re-run (~40k generations, needs a box).
