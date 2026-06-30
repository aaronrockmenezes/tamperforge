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

## P1 — Empirical MAD crux  🔴 GO/NO-GO

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

**Current script:** `experiments/p1_mad_crux.py` with
`--abliterate-layers all` as the default.

**Current blocker before P1:** finish P0 external baselines on Vast:

- DeepSeek V4 Flash full judge for base, HF heretic, HF extreme.
- `lm_eval --model vllm` ARC-Challenge 25-shot full for the same three models.
- Update `docs/results_2026_07_01.md` from the resulting summaries.

---

## P1b — Block vs distributed entanglement

Only if P1 passes. Compare:
- **Block:** single SiLU-MLP adapter at L13 (excisable → T0 only).
- **Distributed:** whole-model post-training pass entangling refusal↔capability
  across layers (no discrete excision point → defends T1).

**Metric:** excision-resistance (can attacker zero a block and keep capability?)
+ capability cost of the procedure + ASR after T0 abliteration.

**Goal:** establish the distributed variant as the real *product*; adapter stays
the fast-iteration prototype.

---

## P2 — Adaptive abliteration (T2)

Attacker recomputes refusal dir as `mean(h|harmful) − mean(h|harmless)` on the
**released entangled model**, abliterates that. Measure ASR + capability.
Tests whether MAD survives a direction derived from our own model rather than
the adapter `W_out` basis.

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

Reproduce D8i cleanly with the P3 harness (how few examples bypass?). Then
attempt adversarial-FT training (TAR-style meta-objective). Honest target:
shift FT cost (e.g. 5 → N examples / more compute), not "solved." Report on the
cost frontier. Position vs TAR + RepNoise.

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
