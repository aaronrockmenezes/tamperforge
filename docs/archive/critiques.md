# Critiques & external review (2026-07-02)

Consolidated external review (friends + ChatGPT, lit-grounded) + our synthesis +
decisions. Read alongside `docs/research_directions_2026_07_02.md` and
`docs/next_steps_2026_07_02.md`.

## Naming scheme (FIX — stop the v5/v6/v7 collision)
Two independent version lines were both using v5/v6/v7. From now on:
- **ABL-v{n}** = abliteration-resistant line → `outputs/tamper_resistant_p1b_v{n}.pt`.
  **ABL-v7** = the current shipped abliteration product (robust across the battery).
- **FTR-v{n}** = fine-tune-resistant line → `outputs/ft_resistant_p4_v{n}.pt`.
  FTR-v2..v5 = done (all = 1-shot moat or worse). **FTR-v6** = LoRA-inner Lever-2 (next).
  **FTR-v7** = scaled version (Blackwell, only if v6 succeeds).
Don't rename existing files (breaks refs); use the ABL-/FTR- prefixes in all talk/docs.

## The external review — verdict: ADOPT ~80%
Core thesis (agreed, matches our data): **frame the paper as anti-cheap-abliteration,
measured by attack-cost frontiers. FT = characterized cost-curve, NOT solved.** v7's
result is the win (clean ASR 0.013/ARC 0.364; attacked v7 ASR 0.004/ARC 0.246;
attacked base ASR 0.662/ARC 0.355 = MAD: attacking base uncensors cheap, attacking
ours collapses capability). Proposed title: *"Cheap Abliteration of Open-Weight LLM
Safeguards Can Be Made Capability-Destructive."*

### Holes it correctly flags (we under-weighted)
1. **Prefill attack — biggest missing test.** Cheap; the "simple attacks" paper shows
   abliteration+prefill drives ASR 10%→16-96%, ART only −10-20%. MUST test.
2. **Harmful-usefulness judge (2nd axis).** ASR alone lies — our k=4 result (ASR 0.72,
   coherent+actionable) proves it. Add a 4-label rubric: refused / harmful-vague /
   harmful-actionable / gibberish. Cheap (re-judge existing gens). Highest value/$.
3. **Extended-refusal / DeepRefusal combo** — cheap, anti-abliteration, composes w/ v7.
4. **HarmBench + BeaverTails** — AdvBench-only is thin.
5. **Attack-cost FRONTIER as the main figure** (not tables): X=attacker budget
   (k dirs / LoRA examples / FT steps), Y=ASR + harmful-usefulness + ARC/MMLU + coherence.
6. **Multi-model** (3 families: Gemma, Qwen, Llama) + **3 seeds** — kills "1B toy."
7. **2nd judge + manual audit** (50-100) + inter-judge agreement for publication.

### Where it's stale (we're ahead)
"P4 FT still remaining" — **done last night**: full-FT frontier K=0→200, v3/v5/v7/base,
ASR+ARC+MMLU, judged. Gap it's right about: that was FULL-FT; attackers use **LoRA/QLoRA**
→ still need a LoRA N×r cost-curve.

### Where it's over-scoped for the next move (defer)
6 variants + WMDP + 3 families + OBLITERATUS-as-benchmark + RepNoise/RMU baselines =
a whole program. Park WMDP-knowledge (big pivot: refusal→hazardous-knowledge) and
OBLITERATUS-as-named-contribution for later.

### Variants it proposes (map to our lines)
- V1 Adapter = mechanism demo (our P1 adapter) — figures only, not the security claim.
- V2 Baked/Distributed = main variant (our ABL baked-weights).
- V3 Ensemble = **ABL-v7** (already have it).
- V4 FT-aware = **FTR-v6/v7** (our Lever-2, now LoRA-inner).
- V5 DeepRefusal = extended-refusal combo (cheap, add later).
- V6 Knowledge = WMDP (deferred).

## DECISION (user, 2026-07-02) — overrides "treat FT as unsolved frontier"
Pursue FT-resistance anyway. Plan:
- **FTR-v6 = LoRA inner** Lever-2 (rank-4, few steps, realistic + fits 4090). Recast of
  the full-param v6. FO-MAML meta-grad + generation-level outer.
- **Eval FTR-v6 on ALL datasets** (AdvBench, HarmBench, BeaverTails, ARC, MMLU, +
  harmful-usefulness judge, prefill).
- **4090 for v6.** Blackwell for **FTR-v7** later, only if v6 shows daylight past K=5.
- Order: **finish FTR-v6, then FTR-v7**, THEN the cheap robustness/evals/multi-seed/
  multi-model battery.

## Key papers (from review + our scan)
Refusal=1 direction (Arditi 2406.11717); shallow safety (2406.05946); simple attacks/
ART (2605.26526); TAR (2408.00761); TamperBench (2602.06911); durability (2412.07097);
SEAM self-destruct (2505.12186); extended-refusal (2505.19056); WMDP (2403.03218);
RepNoise (2405.14577).
