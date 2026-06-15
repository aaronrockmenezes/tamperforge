# MindWeather — Roadmap

> Living doc. Reorder/rewrite as results land. Last updated: 2026-06-14.
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

**Question:** abliterating the refusal direction from the entangled model — does
capability actually drop vs an un-entangled control?

**Setup:**
- Conditions: (a) base, (b) base+entanglement, (c) base abliterated,
  (d) base+entanglement then abliterated.
- Metric: capability (ARC / MMLU subset / PPL) AND refusal/ASR, all 4 conditions.
- Control matters: the gap between (c) and (d) capability is the result.

**Pass:** (d) capability ≪ (c), i.e. abliterating the entangled model costs
significantly more capability than abliterating the bare model — at comparable
ASR. **Fail:** no gap → thesis dead → pivot to "costly-to-abliterate" framing.

**Notes / gotchas (from HANDOFF):**
- Use *inline* L13 abliteration, NOT the misnamed `abliterated_L13/` dir (it is
  all-26-layer).
- `load_advbench()` returns tuples — extract `p[0]`.
- Current prototype adapter is fine for this test (T0 regime).

**Output:** `results/p1_mad_crux.json` + entry in `results.md`.

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
Tests whether MAD survives a direction derived from our own model, not our
SAE-chosen feature list.

**Output:** `results/p2_adaptive_abliteration.json`.

---

## P3 — Evaluation harness (parallel with P2; unblocks trust + scaling)

Replace keyword refusal + tiny-n with rigorous eval. Build as a reusable module
`mindweather/eval/`:
- **ASR:** LLM-judge or HarmBench classifier (not keyword matching).
- **Capability:** ARC-Challenge + MMLU subset + GSM8K.
- **Fluency:** perplexity.
- **Rigor:** n≥100, bootstrap CIs, fixed seeds, config-driven.

All later numbers (and any re-run of P1/P2) use this harness.

---

## P4 — Fine-tuning resistance (T3) — characterize, don't promise

Reproduce D8i cleanly with the P3 harness (how few examples bypass?). Then
attempt adversarial-FT training (TAR-style meta-objective). Honest target:
shift FT cost (e.g. 5 → N examples / more compute), not "solved." Report on the
cost frontier. Position vs TAR + RepNoise.

---

## P5 — Generalize across architectures

Only after P1–P3 hold on Gemma. Qwen3-1.7B, Llama-3.2-1B. No IT SAE → use
mean-difference refusal directions. This is where the "general framework" claim
gets earned.

---

## Parking lot (low priority / future)

- SAE feature Gini coefficient → abliteration-vulnerability proxy (novel, from D7).
- Formalize the cancellation theorem for a write-up.
- Scale to Gemma 3 4B / 9B.
- Qwen adapter retrain with `λ_entangle=1.0` (current 0.099, MAD not established).
- Retrain `safety_adapter_v3_lora_robust.pt` (W_out_align=0.257, broken).
- Emotion-steering Gradio app (separate thread, complete research).

## Codebase cleanup (do AFTER P1 — don't polish an unverified thesis)

- `setup.sh` — assert `env_ml`, check HF auth, verify deps, smoke-test SAE load.
- Device helper (cuda/mps/cpu) in package — one place, not N scripts.
- Consolidate duplicated abliteration/refusal logic into the package; scripts
  become thin CLIs (revises the old "standalone scripts" rule — a framework
  needs a source-of-truth package).
- Config-driven hyperparams (`configs/*.yaml`), kill CLI flag soup.
- `ruff` + pinned seeds.
- Rename/regenerate the misnamed `abliterated_L13/`.
