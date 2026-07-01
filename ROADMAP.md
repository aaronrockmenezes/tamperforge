# tamperforge — Roadmap

> Living doc. Reorder/rewrite as results land. Last updated: 2026-07-01.
> Read `THREAT_MODEL.md` first — it defines the attacker tiers (T0–T3) and the
> success metric (attacker cost to reach X% ASR at ≤Y% capability loss).

## Sequence

```
P0 → P1 → P1b → (P3 ∥ P2) → P4 → P5
```

Each phase gates the next. Do not advance on un-verified foundations. P1 is
go/no-go for the entire thesis.

---

## P0 — Threat model & attacker tiers  ✅ (this doc + THREAT_MODEL.md)

Deliverable: `THREAT_MODEL.md`. Attacks tiered T0–T3, metric defined.
Status: **done** — revise as understanding sharpens.

---

## P1 — Empirical MAD crux  🟡 POC PASS (mechanism), ceiling found

**Question:** after training an adapter on clean Gemma to couple refusal behavior
to language/capability directions, does ablating the adapted model's adapter
`W_out` directions cost more capability than bare Gemma refusal-direction
ablation?

**Setup:**
- Conditions: base, base_ablated, base_adapter,
  base_adapter_ablated_full, base_adapter_ablated_adapter_only.
- Metric: capability (ARC / MMLU subset / PPL) AND refusal/ASR, all 5
  conditions.
- Control matters: compare bare Gemma refusal-direction ablation against full
  adapted-model `W_out` ablation at comparable ASR.

**Pass:** adapted model keeps pre-attack capability close to base, and adapted
full ablation damages capability more than bare-model ablation at comparable
ASR. **Fail:** no gap → thesis dead → pivot to "costly-to-abliterate" framing.

**Notes / gotchas (from HANDOFF):**
- Future tamperforge experiments use all-layer abliteration. L13-only is a
  one-time legacy comparison.
- P1 must not rely on SAE directions. Use empirical refusal direction for bare
  Gemma and adapter `W_out` directions for adapted-model attacks.
- Do not reuse the old mindweather `abliterated_L13/` dir; it was suspect.
- `load_advbench()` returns tuples — extract `p[0]`.
- Current prototype adapter is fine for this test (T0 regime).

**Output:** `results/<run_id>/summary.json`, `generations.jsonl`, `events.jsonl`,
and optional `judgments.jsonl`.

**Current script:** `experiments/p1_mad_crux.py` (6 conditions incl the
`base_ablated_randN` direction-count control, `--conditions` subset flag,
`--adapter-attack-rank` sweep, `--adapter-base native_ablated`).

**RESULT (2026-07-01):** POC pass on the *mechanism*, ceiling on the *defense*.
- P0 done: cheap uncensoring is ~free today (heretic judge ASR 0.886 at ARC ≥ base).
- Clean-base rank sweep: entanglement PROVEN — ablating adapter W_out dirs drives
  PPL 14.8→9513 vs random dirs 13.5→24.5 (~400× at k=32). Confound dead. Safety
  never dropped (native refusal masks it on clean base) → mechanism, not defense.
- Abliterated-base sweep (`--adapter-base native_ablated`, the product): crossover
  at k=4 — ablating the entangled adapter uncensors (judge ASR 0.72) at capability
  cost (PPL 13→23.5, ARC 0.43→0.34) while the bare attack uncensors free and
  random-k is untouched. That is the MAD property.
- **Ceiling:** the attacker's crack point (k=4) yields COHERENT, actionable
  harmful output — a usable model. Prose-PPL rises but generations stay fluent
  (prose-PPL ≠ generation coherence). So the adapter proves "uncensoring costs
  capability" but does not yet yield "uncensoring gives a useless model." That
  gap is P1b's job. Full curve + tables in `docs/results_2026_07_01.md`.

---

## P1b — From removable adapter to un-excisable, gibberish-on-attack  🟢 ROBUST POC PASS (A/v7)

Two prototype weaknesses to close (see `docs/p1b_plan.md`):
1. adapter is a removable block (attacker deletes it) → fold into weights (B), or
   train the base directly (A).
2. attack yields coherent uncensored output → make ablation collapse GENERATION,
   not just prose-PPL.

**A — adversarial tamper-resistance (primary).** `experiments/train_tamper_resistant.py`.
Differentiable ablated forward via `torch.func.functional_call` (validated on a
toy: grads flow, clean-good/ablated-wrecked). Trains base MLP weights so ablating
the EMPIRICAL refusal direction both removes safety and destroys capability.
Loss = L_task(clean) + λ_safe·L_safe(clean refuses) + λ_gib·relu(gap_target −
(L_abl − L_task)) [selectivity] + λ_uncensor·relu(margin − refusal_loss(ablated))
[couple safety to d] + λ_reg·drift.

Iteration log (2026-07-01):
- Absolute-badness gib loss → model self-destructs (both losses rise together).
  Fixed with GAP-based loss (reward ablated worse than clean, not absolute).
- Gap opens with a hard push (λ_gib 8: held-out gap +18, L_task healthy) but on a
  SAFETY-IRRELEVANT direction — ablated model still refuses. Added L_uncensor to
  force refusal onto d.
- v5 (600 steps, MLP, prose-PPL gib): FAILED — trained_attacked judge-ASR 0.60 ≈
  base 0.54. Prose-PPL objective ruled out (prose PPL ≠ harmful-gen prevention).
- **v6 (400 steps, `--gib-mode argmax` + `--train-scope all`): POC PASS.** The
  argmax gib loss (ablated model must fail to reproduce the CLEAN model's own
  greedy generations) broke generation on attack. Full AdvBench 500 + lm_eval
  ARC-25:

  | condition | judge ASR | ARC acc |
  |---|---:|---:|
  | trained_clean | 0.010 | 0.362 |
  | trained_attacked | 0.032 | 0.294 |
  | base_attacked | 0.844 | 0.355 |

  Same attack: 0.844 ASR on base → 0.032 on trained (~26×). Ablating the trained
  model yields non-harmful gibberish + ARC drop; clean product ≈ free (ARC 0.362
  vs base 0.366). Checkpoint `outputs/tamper_resistant_p1b_v6.pt` (git+HF). Detail
  in `docs/results_2026_07_01.md` / devlog part 3.
- **v7 (`--attack-ensemble`): ROBUST POC PASS.** Randomizing the simulated attack
  (scope × layers × direction prompts) each step closed every v6 leak — mlp-only
  0.22→0.013, L13-25 0.11→0.004, seed7 0.11→0.000, matched 0.004 — clean product
  intact (ASR 0.013, ARC 0.364, gen parse 1.0). Attacking v7 collapses ARC to
  random (0.246) while attacking base keeps it (0.355): MAD on both axes. Full
  battery + tables in `docs/results_2026_07_01.md`, `docs/attack_battery.md`.
- **NEXT (POC → publishable), in priority:**
  1. **Fine-tuning attack (P4)** — the realistic threat; abliteration-robust ≠
     FT-robust. Highest priority. Build `experiments/ft_attack.py`.
  2. **OBLITERATUS (P2)** — adaptive attacker (per-layer, entanglement-gated
     skipping). Separate AGPL harness on the saved v7 dir.
  3. Rigor: seeds (is v7 a lucky run?), MMLU-full + GSM8K, Qwen3-1.7B/Llama-3.2-1B;
     ablate which of {argmax, attention scope, λ_uncensor, ensemble} is load-bearing.
  4. Note: the SVD/whitened attack estimators are duds (don't uncensor base) — a
     real stronger-estimator test needs a better refusal-SVD; low priority.

**B — fold adapter into FFN (cheap, un-block).** `tamperforge.GatedSafetyAdapter`
(SwiGLU, foldable) + `fold_gated_adapter_into_ffn` + `verify_fold`. Ready; needs a
gated-adapter retrain. Removes the discrete block but base stays coherent, so B
alone won't give gibberish — pairs with A.

**C — weight-space rotation (geometric sanity).** Rotate the empirical refusal
direction into the top capability singular directions so ablating it deletes
capability. Specced, not built.

**Metric:** excision-resistance + capability cost of the procedure + (the new bar)
does a successful abliteration produce an incoherent model, verified on
GENERATIONS not prose-PPL, and does it survive OBLITERATUS (P2 attacker).

---

## P2 — Adaptive abliteration (T2)

Attacker recomputes refusal dir as `mean(h|harmful) − mean(h|harmless)` on the
**released entangled model**, abliterates that. Measure ASR + capability.
Tests whether MAD survives a direction derived from our own model rather than
the adapter `W_out` basis.

**Attacker tool:** `elder-plinius/OBLITERATUS` (whitened-SVD, per-layer,
cluster-aware layer selection, entanglement-gated skipping, Ouroboros self-repair
detection) — purpose-built to dodge exactly this defense. Use it as a SEPARATE
harness (AGPL-3.0: pip-install + run on our model; never vendor its code). If the
entanglement survives an attacker that actively avoids entangled layers, the
claim holds.

**Output:** `results/p2_adaptive_abliteration.json`.

---

## P3 — Evaluation harness (parallel with P2; unblocks trust + scaling)

Replace keyword refusal + tiny-n with rigorous eval. Build as a reusable module
`tamperforge/eval/`:
- **ASR:** LLM-judge or HarmBench classifier (not keyword matching).
- **Capability:** ARC-Challenge + MMLU subset + GSM8K.
- **Fluency:** perplexity.
- **Rigor:** n≥100, bootstrap CIs, fixed seeds, config-driven.

All later numbers (and any re-run of P1/P2) use this harness.

**Current harness:** `src/tamperforge/eval/` and runbook `docs/eval_runbook.md`.

---

## P4 — Fine-tuning resistance (T3) — characterize, don't promise

**Attack characterized (2026-07-01): NEGATIVE for v7.** FT of v7 on K=1 harmful
demo restores coherent-harmful (judge ASR 0.013→0.740), tracking the base control
(→0.800) at every K∈{1,5,10,25}. Abliteration-robust ≠ FT-robust: the MAD trap
only fires on ablation, FT retrains around it. Table in
`docs/results_2026_07_01.md` (P4 section).

**Now the active thread (user priority): make the method FT-resistant.** Not
tamper-proof — shift the frontier so it doesn't break at K=1. Plan in
`docs/ft_resistance_plan.md`: TAR-style meta-training (simulate attacker FT in the
inner loop, shape weights so the FT'd model stays useless-on-harm; reuse P1b-A
functional_call plumbing + argmax-divergence collapse loss + ensemble-over-attackers),
optionally + RepNoise-style harmful-rep scrambling. Metric = re-run the exact P4
sweep; go/no-go K=1 ASR < 0.20 at ARC ≥ 0.30. Mandatory adaptive-attacker sweep
(vary lr/epochs/held-out demos) to rule out gradient-masking. Position vs TAR +
RepNoise. OBLITERATUS (P2) deferred behind this.

---

## P5 — Generalize across architectures

Only after P1–P3 hold on Gemma. Qwen3-1.7B, Llama-3.2-1B. Use empirical
mean-difference refusal directions plus learned adapter directions. This is
where the "general framework" claim gets earned.

---

## Parking lot (low priority / future)

- SAE feature Gini coefficient → abliteration-vulnerability proxy (novel, from D7).
- Formalize the cancellation theorem for a write-up.
- Scale to Gemma 3 4B / 9B.
- Qwen adapter retrain with `λ_entangle=1.0` (current 0.099, MAD not established).
- Retrain `safety_adapter_v3_lora_robust.pt` (W_out_align=0.257, broken).
- Emotion-steering Gradio app (separate thread, complete research).

## Codebase cleanup (do AFTER P1 — don't polish an unverified thesis)

- `setup.sh` — assert `env_ml`, check HF auth, verify deps, smoke-test base load.
- Device helper (cuda/mps/cpu) in package — one place, not N scripts.
- Consolidate duplicated abliteration/refusal logic into the package; scripts
  become thin CLIs (revises the old "standalone scripts" rule — a framework
  needs a source-of-truth package).
- Config-driven hyperparams (`configs/*.yaml`), kill CLI flag soup.
- `ruff` + pinned seeds.
- Keep generated local ablated checkpoints under `outputs/` with
  `abliteration_meta.json`.
