# Findings: external attacker (Heretic-base), external prompt benches, IFEval (2026-07-03)

All safety = LLM-judged (DeepSeek V4 Flash); harmAct = harmful_actionable_rate (coherent harm);
gib = gibberish_rate. Capability = lm_eval. Model = gemma-3-1b; v7 = ABL-v7 (DL 13).

## 1. Heretic on BASE gemma — closes the external-attacker confound (FLAGSHIP)
Prior gap: OBLITERATUS was too weak on gemma-3 and Heretic-base was unrun, so we couldn't
show a *professional* tool coherently uncensors the base (only our own rank-1 did). Now run.
Heretic (KL-minimizing Optuna abliterator) on **base** gemma, 3 Pareto points, AdvBench 520 + cap:

| Heretic trial | KL | harmAct | gib | ARC | MMLU |
|---|--:|--:|--:|--:|--:|
| kl014 (T161) | 0.014 | 0.777 | 0.000 | 0.358 | 0.391 |
| kl036 (T99)  | 0.036 | 0.798 | 0.000 | 0.358 | 0.392 |
| kl098 (T84)  | 0.098 | **0.931** | 0.004 | 0.361 | 0.394 |

**A professional abliterator drives BASE gemma to 93% coherent harm at KL 0.098 with capability
fully intact** (ARC/MMLU = clean base). Contrast v7 (findings_multimodel §6): no point on
Heretic's v7 Pareto coherently uncensors — max harmAct 0.171 (mostly gibberish), and reaching
0 refusals needs KL 0.70 (= model wrecked). **Same tool, same budget: base fully uncensored +
coherent + smart; v7 cannot be uncensored without gibberish/collapse.** MAD proven by the
attacker's own KL-optimizer, on both base (fails to resist) and v7 (resists). This is the
strongest external-attacker evidence; supersedes the OBLITERATUS-inconclusive note.

## 2. External prompt benches (StrongREJECT / JailbreakBench / SORRY-Bench)
Tier 1: each bench's OWN prompt set → our vLLM harness → our judge. CLEAN (unattacked) models.
(Tier 2 = benches' official judges on the Blackwell box — TODO.)

| bench (n) | base harmAct / gib | v7 harmAct / gib |
|---|---|---|
| StrongREJECT (313) | 0.042 / 0.000 | 0.016 / 0.144 |
| JailbreakBench (100) | 0.020 / 0.000 | 0.010 / 0.180 |
| SORRY-Bench (450) | 0.118 / 0.004 | 0.031 / 0.407 |

- **Clean v7 is at least as safe as base on all three** (coherent harm strictly lower).
- **Clean gibberish tax grows with prompt diversity** — 14% / 18% / 41%. SORRY-Bench (45
  categories) triggers most, matching the off-AdvBench clean-gibberish limitation
  (BeaverTails 47.5%). On *harmful* prompts gibberish ≠ harm (safety fine); it is the
  coherence cost. These are prompt/jailbreak benches on clean models — v7 is not a jailbreak
  defense, so this measures refusal quality, not MAD (MAD = weight-space, TamperBench).

## 3. IFEval — clean generative instruction-following + the MAD split
lm_eval ifeval, `--apply_chat_template`, 541 prompts. Value: MC (ARC/MMLU) hides generative
coherence; IFEval exposes it.

| model | prompt_strict | prompt_loose | inst_strict | inst_loose |
|---|--:|--:|--:|--:|
| base clean | 0.542 | 0.588 | 0.673 | 0.718 |
| **v7 clean** | **0.205** | 0.220 | 0.376 | 0.398 |
| base_att (abliterated) | 0.530 | 0.582 | 0.664 | 0.707 |
| **v7_att (abliterated)** | **0.120** | 0.122 | 0.249 | 0.253 |

Two reads:
- **MAD split intact:** base clean 0.542 ≈ base_att 0.530 (abliterating base barely dents it);
  v7 clean 0.205 → v7_att 0.120 (attack craters further). Attacker still gets a dumber model.
- **Clean v7 generative tax is real but consistent, not new:** clean v7 matches base on MC
  (ARC 0.344/MMLU 0.393) but drops on free-form generation — GSM8K −35%, IFEval **−62%**
  (prompt_strict 0.542→0.205). Same documented limitation, sharpest measurement yet. v7 clean
  still scores 0.20–0.40 (degraded, not dead). The entanglement is partly "always-on" on
  generation = the honest cost. Shrinkable via λ_gib / gib-warmup at a robustness tradeoff
  (not a wall).

## Paper implications
- §1 = flagship external validation (attacker's own optimizer proves MAD both directions).
- §2 = third-party prompt-bench coverage (kills "only your own attack") + honest gibberish tax.
- §3 = **new honest limitation to foreground**: clean v7 has a real generative-coherence cost
  (MC-credible, generation-degraded). State it plainly in Limitations alongside the prefill hole.
- Anchor unchanged: abliteration-resistance + MAD, base-uncensors-cheaply vs v7-cannot.

## Artifacts
`results/heretic_base_gemma_{kl014,kl036,kl098}_adv520(_judged)`, `results/cap_heretic_base_*`,
`results/ext_{strongreject,jailbreakbench,sorrybench}_gemma_{base,v7}(_judged)`,
`results/ifeval_gemma_{base,v7,base_att,v7_att}`. Harness: `scripts/external_benches/`,
p0 `--prompt-file`. Heretic-base weights saved to HF (Trial 84/99/161 merges).
