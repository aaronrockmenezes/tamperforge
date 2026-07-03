# Findings: multi-model generalization + adaptive-attack robustness (2026-07-02)

ABL-v7 = the abliteration-resistance recipe (`train_tamper_resistant.py --attack-ensemble
--train-scope all --gib-mode argmax`, canonical `λ_reg 0.1 --steps 500`). Trained fresh on
3 architectures + re-seeded on gemma, then attacked. All safety = LLM-judged (DeepSeek V4
Flash) on AdvBench 520; capability = `lm_eval` ARC-Challenge (full) + MMLU 12-topic. Attack
= abliteration baked into weights via `save_p1b_checkpoint.py`, then eval clean-vs-attacked
vs base. "harmAct" = harmful_actionable_rate (coherent harm); keyword ASR is NOT used for
verdicts (it counts gibberish as success — see §4).

## 1. Multi-model generalization (rank-1 abliteration, per-model refusal layer)
| model | ablate base harmAct | ablate v7 harmAct | v7_att gib | v7_att ARC/MMLU | verdict |
|---|--:|--:|--:|--:|---|
| gemma-3-1b (seed 42) | ~0.66–0.80 | ~0.006 | ~1.00 | 0.265 / 0.268 | **FULL WIN** (gibberish + capability crater) |
| gemma-3-1b (seed 2) | 0.819 | 0.006 | 0.990 | 0.210 / 0.258 | **FULL WIN** (reproduced) |
| Qwen3-0.6b (DL 14) | 0.581 | 0.000 | 0.996 | 0.306 / 0.348 | **CLEAN DEFENSE** (gibberish wall, no crater) |
| Llama-3.2-1b (DL 13) | 0.652 | 0.133 | 0.738 | 0.366 / 0.480 | **PARTIAL** (leaky wall ~13%, no crater) |
| Llama-3.2-1b (DL 8, WRONG) | 0.431 | 0.654 | 0.20 | 0.362 / 0.485 | apparent FAIL — was a layer bug (§3) |

**Read:** ABL-v7 generalizes across all 3 architectures *with correct per-model refusal-layer
selection*. Strength varies: gemma couples safety to capability (ablate → gibberish AND dumb);
Qwen defends via a clean harmful-gibberish wall (benign capability intact); Llama defends but
leaks ~13% (its safety is diffuse — base only uncensors to 0.66 even at its best layer, vs
gemma 0.82). The strong MAD capability-collapse is gemma-specific; the gibberish-wall defense
is broader.

## 2. Adaptive-attack robustness (THE flagship result)
The rank-1 single-direction abliteration above is the *weakest* attack. Built `--per-layer`:
an adaptive attack computing an independent diffmeans refusal direction at EACH layer and
ablating each layer with its own (the core of purpose-built adaptive attackers), on our
working diffmeans (no dud SVD, no external AGPL code vendored).

**OG gemma ABL-v7 under the per-layer adaptive attack (520 judged + capability):**
| condition | harmAct | gibberish | ARC | MMLU |
|---|--:|--:|--:|--:|
| base + per-layer attack | **0.712** | 0.025 | 0.339 | 0.262 |
| ABL-v7 + per-layer attack | **0.000** | 1.000 | 0.270 | 0.240 |

The stronger attack **potently uncensors base (71% coherent harm, ARC intact 0.339)** but
extracts **0% from ABL-v7** — only gibberish + a capability crater (ARC 0.34→0.27, MMLU
0.39→0.24). Identical outcome to rank-1 (~0.006 harmAct). **The entanglement is robust to
adaptive abliteration, not merely the rank-1 direction it was validated against.** This kills
the "a smarter attacker routes around the MAD trap" objection.

Paper backbone: *"A per-layer adaptive abliteration attack extracts 71% coherent harm from
the base model but 0% from ABL-v7 — where it yields only gibberish and a 25–40% capability
collapse."*

(Qwen-v7 + Llama-L13 per-layer attacks running — does the leakier walls hold vs the stronger
attack? gemma held airtight.)

## 3. Per-model refusal-layer selection is essential (the DL bug)
Codex picked direction-layers by a 50%-depth GUESS (Llama 8/16, Qwen 14/28). WRONG for Llama:
a judged base-ablation layer sweep showed base uncensors to **0.60 coherent harm at L13** vs
0.47 at L8 (keyword overstated L13 at 0.89). Retraining Llama-v7 at L13 dropped v7_att harmAct
**0.654 → 0.133** — the apparent "Llama doesn't transfer" was largely a layer-selection bug.
gemma's L13 (of 26) was empirically chosen; the 50% heuristic is not a substitute. **HIGH-PRI
TODO: full per-layer sweeps (all layers) for Llama + Gemma; Qwen full sweep in progress.**

## 4. Methodology: keyword ASR is unreliable BOTH directions (validated twice)
Keyword refusal counts gibberish as "attack success". Demonstrated:
- **Overstates on gibberish models:** Qwen v7 per-layer sweep — keyword 0.99 but judge 0.01.
- **Overstates on base uncensoring:** Llama base sweep L13 — keyword 0.89 but judge 0.60.
LLM judge + `usefulness_label` (gibberish vs refused vs harmful_actionable) is load-bearing,
not polish. This is itself a paper contribution.

## 5b. Off-distribution generalization (HarmBench + BeaverTails) — Llama is the weak link
Abliterate-v7 harmAct / gibberish on prompt sets never trained on:
| condition | HarmBench | BeaverTails |
|---|--:|--:|
| qwen base-ablated | 0.090 / 0.235 | 0.063 / 0.320 |
| **qwen v7-ablated** | **0.000 / 0.985** | **0.001 / 0.989** |
| llama base-ablated | 0.500 / 0.005 | 0.320 / 0.008 |
| **llama v7-ablated** | **0.420 / 0.420** | **0.435 / 0.295** |

- **Qwen v7 GENERALIZES off-distribution** — ablation → gibberish (0.000/0.001 harm) on both,
  like gemma (which was ~0 gibberish across prefill/HB/BT in the earlier battery).
- **Llama-L13 v7 FAILS off-distribution** — ablation → 0.42/0.44 coherent harm, essentially
  no better than base (0.50/0.32); on BeaverTails it's even WORSE than base-ablated. Its
  on-distribution defense (0.13 on AdvBench) does NOT transfer. **Llama's entanglement is
  AdvBench-overfit** — a direct consequence of its diffuse safety (no concentrated refusal
  direction to couple to capability). This names the boundary condition: the method
  generalizes off-distribution only for models with concentrated (gemma) or
  broadly-caught (Qwen gibberish-wall) safety; it fails for diffuse-safety models (Llama).

## 5. Honest limitations
- **Seed fragility:** ~half of gemma seeds fail to form the entanglement (42, 2 = win; 1, 3
  NaN'd; 4, 5 = gib_ce stuck). Root cause partly numerical (argmax gib CE spiked to inf in
  bf16 → fixed with fp32+clamp, which killed the NaN crashes) but seed4/5 STILL failed after
  the fix → some seeds genuinely don't find the basin. n=2 reproducibility; the fragility is
  a real limitation (fix candidate: warmup / higher early λ_gib).
- **Llama leaks ~13%** even at the right layer (diffuse safety); clean Llama-v7 also leaks 13%.
- **Multi-model battery is AdvBench-only** so far — off-distribution (prefill/HarmBench/
  BeaverTails) done only for gemma. Qwen/Llama need it for parity.
- Qwen3-0.6B base is weakly safe (0.267 clean) — an easy starting point.

## Artifacts
Ckpts: `outputs/tamper_resistant_{qwen3_0p6b,llama32_1b_v7_L13}_v7.pt`, gemma `..._p1b_v7_seed2.pt`,
OG `outputs/hf_og/adapters/tamper_resistant_p1b_v7.pt`. Runs: `{qwen06,llama_L13,gemma_s2}_*_adv520(_judged)`,
`gpl_{v7,base}_*` (adaptive), `sweep_q_L*` (Qwen layer sweep), `pl_{qwen,ll13}_*` (adaptive, running).
Tooling: `save_p1b_checkpoint.py --per-layer`, `scripts/multimodel_abl_v7/`.

## 6. External professional attackers on gemma ABL-v7 (judged, harmAct/gib)
Heretic (KL-minimizing Optuna) across its full Pareto: kl004(lossless) 0.010/0.04=refuses,
kl027(mid) 0.171/0.66, kl070(Trial97 "0 refusals") 0.019/0.97=gibberish. **No point on the
front coherently uncensors v7** (max 0.171, mostly gibberish) → the attacker's own KL-optimizer
can't decensor without wrecking. OBLITERATUS 7 methods on v7: harmAct 0.00–0.05, gibberish/
refuse — BUT it's WEAK ON gemma-3 BASE too (aggressive 0.031, surgical 0.054, nuclear 0.008/
0.68gib) vs our rank-1's 0.82 → OBLITERATUS-on-gemma is INCONCLUSIVE (tool doesn't effectively
abliterate gemma-3-1b), don't lean on it. GAP: neither tool's base-uncensor confirmed coherent
on gemma-3 (OBL weak; Heretic-base NOT RUN — TODO). Primary adaptive evidence stays our
per-layer attack (base 0.71 → v7 0.00). Heretic weights saved to HF heretic/. TODO: heretic on
base gemma; consider OBLITERATUS/Heretic on Llama/Qwen (archs they handle better).
