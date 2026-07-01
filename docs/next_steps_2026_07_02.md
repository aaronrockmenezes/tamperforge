# Next steps — phased plan (2026-07-02)

Frozen decisions. Resume from here next session (fresh box).

## Phase A — FT-resistance, one honest shot (Lever 2, Patcher/TAR-done-right)
Fix the crux: inner sim must MATCH the real attack on GENERATION.
- Real **AdamW** inner loop (momentum), multi-epoch, **all-params** (not first-order
  SGD, not just 182 matrices) — mirrors `ft_attack`. Patcher-style extended/parallel
  adversary. Generation-level outer objective (v4/v5 machinery, now engaging).
- Warm-start v7 (keep abliteration robustness). Re-sweep judge-ASR vs
  **K ∈ {1,5,10,25,50,100}** at **full AdvBench 520**.
- **STOP RULE: 3–5 runs. If it does NOT get close to SOTA (resist to K in the
  dozens at ARC/MMLU ≥ 0.30) → KILL FT chasing.** Do not grind past 5 runs.

## Phase B — if FT killed (or done): rigor / scale-up
- **Seeds:** 3 total (2 more) on the headline result — is it a lucky run?
- **Models:** 5 total (4 more) — gemma-3-1b + Qwen3-1.7B + Llama-3.2-1B + 2 more
  (e.g. gemma-3-4b, Qwen3-4b). **3 seeds each → 15 variants.**
- **Full eval on everything** (push to full after FT dropped/done):
  - Abliteration battery (the v7 battery) on all 15.
  - FT frontier (whatever Phase A reached) on all 15.
  - Capability: ARC + **MMLU-full + GSM8K**, clean AND post-attack.
  - **Compose check:** does the FT defense preserve v7 abliteration-robustness? Both
    axes on ONE model — the combined artifact is the product.
- **Adopt TamperBench** as the eval harness (external/standardized > home-grown) +
  **ART head-to-head** (abliteration baseline) + **jailbreak-tuning** (strongest attack).

## Phase C — write-up
Anchor: **abliteration-resistance + MAD mechanism** (novel, proven: P1 ~400× PPL) +
**honest FT cost-frontier**. Framing: "smart-and-safe XOR dumb-and-dangerous."
Do NOT gate the paper on solving FT (unsolved by the field).

## Venue targets
- **Workshop first (fast, right audience, fits current+scaled result):**
  NeurIPS 2026 safety workshop (Safe Generative AI / Red-Teaming) or SoLaR. Deadlines
  ~Sept 2026 (VERIFY exact dates). Plant the flag, get feedback.
- **Main track (ICLR 2027, deadline ~Sept 2026 — VERIFY):** only with the full
  15-variant + FT-frontier + TamperBench + ART baseline. MAD mechanism = the
  main-track novelty hook. 1B/1-seed alone = desk-reject risk.
- Sequence: workshop now → main conf when Phase B lands.

## Moonshot (only if Phase A plateaus, deferred)
Loss-landscape basin-trap / capability moat — see
`docs/research_directions_2026_07_02.md`.
