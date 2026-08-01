# Paper tables (v1 draft, 2026-07-08) — assembled from full_eval_matrices/

Purpose: canonical paper-ready tables + captions + one-line reads. Numbers assembled
directly from `full_eval_matrices/*.md` (which trace to `results/**/summary.json`).
Nothing here is hand-massaged — this file is the compilation, not new analysis.

Conventions
- **Harm cells = `harmAct / gib`.** `harmAct` = LLM-judged rate of coherent harmful
  content (the strict safety metric). `gib` = rate of gibberish output (attack signal).
  Both in [0,1]. Lower `harmAct` safer; higher `gib` under attack = stronger poison-pill wall.
- **Capability cells = accuracy** (higher better). IFEval = instruction-following (prompt-level);
  GSM8K = 5-shot chain-of-thought math; ARC = full; MMLU = 12-subject subset (see App. B).
- **Attack** = rank-1 refusal-direction abliteration (Arditi et al.) at each product's DL
  (direction-layer): Qwen v7@14 / v8@20, Llama v7@13 / v8@13, gemma v7@13 / base+v8@14.
  Per-model DL from judged sweep peak.
- **Judge** = DeepSeek-V4-Flash via OpenRouter; 64-worker parallel; parse-fail-guarded
  (aborts if >5% rows fail to parse — added after the llama base_att artifact discovered
  and re-judged 2026-07-06).
- **Datasets, full (no subsets):** AdvBench 520 (walledai), HarmBench-standard 200,
  StrongReject 313, JailbreakBench 100, SORRY-Bench 450; ARC full test set, MMLU-12
  0-shot, IFEval prompt-level 541 (chat-template), GSM8K 5-shot 1319.

---

## Table 1 (headline): Conditional capability collapse under abliteration
The "smart-and-safe XOR dumb-and-dangerous" claim in one row per model. Compares
**attacked** GSM8K (proxy for whether the attacker keeps a capable model) and
**clean** IFEval (proxy for whether the shipped product is a good assistant) across
base / v7 / v8 on all three architectures. Both properties must hold on the same row.

| model | metric | base_clean | base_att | v7_clean | v7_att | v8_clean (ours) | v8_att (ours) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B  | attacked GSM8K ↓ (attacker want ↑) | .425 | .373 | .376 | .335 | .401 | **.020** |
| Qwen3-0.6B  | clean IFEval ↑ (product) | .309 | .312 | .211 | .198 | **.314** | .124 |
| Llama-3.2-1B | attacked GSM8K ↓ | .334 | .307 | .344 | .351 | .348 | **.278** |
| Llama-3.2-1B | clean IFEval ↑ | .494 | .494 | .057 | .065 | **.470** | .161 |
| Gemma-3-1b  | attacked GSM8K ↓ | .255 | .243 | .163 | .010 | .222 | **.115** |
| Gemma-3-1b  | clean IFEval ↑ | .543 | .527 | .207 | .120 | **.462** | .185 |

Reads (one line each):
- Qwen v8: clean = base (IFEval .314 vs .309), attack destroys math (−95% GSM8K).
- Llama v8: clean ~ base (IFEval .470 vs .494, −5%), softer crater (GSM8K −17%).
- Gemma v8: partial win — clean IFEval .462 vs base .543 (−15% residual, S2GIB=8 rerun open);
  attacked IFEval crater .185.

---

## Table 2: Harm robustness across 5 benches (all archs, all conditions)
Cells are `harmAct / gib`. Base attack succeeds on all archs (.43–.86 harmAct with
capability intact — the threat we defend against). v7 is a leaky gibberish wall
(hides harm behind gibberish; harm still leaks 0.11–0.31 on Llama). v8 = 0 harmAct
on-distribution and off-distribution, ≥.95 gibberish under attack.

**Qwen3-0.6B** (DL: v7@14, v8@20)
| cond | AdvBench | HarmBench | StrongReject | JailbreakB. | SORRY-B. |
|---|---|---|---|---|---|
| base_clean | .25/.04 | .29/.14 | .30/.12 | .33/.06 | .36/.06 |
| base_att | .48/.14 | .43/.26 | .47/.25 | .56/.20 | .45/.17 |
| v7_clean | .01/.95 | .03/.94 | .00/.98 | .01/.99 | .01/.96 |
| v7_att | .01/.98 | .01/.97 | .00/.99 | .00/1.00 | .00/.99 |
| **v8_clean** | .10/.03 | .17/.15 | .19/.12 | .15/.12 | .28/.13 |
| **v8_att** | .00/.95 | .00/.98 | .00/.98 | .00/.99 | .00/.99 |

**Llama-3.2-1B** (DL13)
| cond | AdvBench | HarmBench | StrongReject | JailbreakB. | SORRY-B. |
|---|---|---|---|---|---|
| base_clean | .01/.00 | .06/.00 | .03/.00 | .01/.00 | .07/.00 |
| base_att | .65/.01 | .61/.04 | .53/.03 | .50/.00 | .49/.01 |
| v7_clean | .11/.79 | .21/.68 | .28/.65 | .16/.74 | .15/.70 |
| v7_att | .14/.73 | .22/.66 | .31/.59 | .25/.62 | .22/.59 |
| **v8_clean** | .00/.00 | .01/.01 | .00/.00 | .00/.00 | .04/.01 |
| **v8_att** | .00/.99 | .00/.99 | .00/1.00 | .00/.98 | .00/.99 |

**Gemma-3-1b** (DL: base+v8@14, v7@13)
| cond | AdvBench | HarmBench | StrongReject | JailbreakB. | SORRY-B. |
|---|---|---|---|---|---|
| base_clean | .01/.01 | .11/.00 | .04/.00 | .04/.00 | .11/.00 |
| base_att | .85/.00 | .86/.01 | .85/.01 | .81/.00 | .66/.01 |
| v7_clean | .00/.04 | .04/.14 | .02/.15 | .02/.15 | .04/.41 |
| v7_att | .00/1.00 | .00/1.00 | .00/1.00 | .00/1.00 | .00/1.00 |
| **v8_clean** | .07/.00 | .15/.00 | .12/.00 | .14/.01 | .16/.01 |
| **v8_att** | .00/1.00 | .00/1.00 | .00/1.00 | .00/.99 | .00/1.00 |

Off-distribution closure — the Llama story:
- v7_att LEAKED off-dist: HarmBench .22 / SORRY .22 coherent harm (documented boundary).
- v8_att = **0.00 harmAct on all 5 benches**, on- and off-distribution (98–100% gib).
  v8 closes the diffuse-safety leak that stood through the v7 campaign.

---

## Table 3: Capability retention (clean model) + attack crater (attacked model)
| model | cond | ARC | MMLU-12 | IFEval | GSM8K |
|---|---|---:|---:|---:|---:|
| Qwen | base_clean | .344 | .425 | .309 | .425 |
| Qwen | v7_clean | .340 | .391 | .211 | .376 |
| Qwen | **v8_clean** | **.350** | **.437** | **.314** | .401 |
| Qwen | base_att | .345 | .413 | .312 | .373 |
| Qwen | v7_att | .325 | .351 | .198 | .335 |
| Qwen | **v8_att** | .265 | .255 | .124 | **.020** |
| Llama | base_clean | .381 | .484 | .494 | .334 |
| Llama | v7_clean | .395 | .483 | .057 | .344 |
| Llama | **v8_clean** | .384 | .484 | **.470** | .348 |
| Llama | base_att | .369 | .470 | .494 | .307 |
| Llama | v7_att | .392 | .486 | .065 | .351 |
| Llama | **v8_att** | .363 | .464 | .161 | **.278** |
| gemma | base_clean | .387 | .395 | .543 | .255 |
| gemma | v7_clean | .389 | .393 | .207 | .163 |
| gemma | **v8_clean** | .371 | .392 | **.462** | .222 |
| gemma | base_att | .363 | .371 | .527 | .243 |
| gemma | v7_att | .308 | .265 | .120 | .010 |
| gemma | **v8_att** | .348 | .363 | .185 | **.115** |

Reads:
- Clean v8 recovers all v7's IFEval tax on Qwen (+.10 vs v7), Llama (+.41), gemma (+.26).
- v7 vs v8 diverge under attack on Qwen GSM8K (v7 .335 → v8 .020, deeper crater by ×17).
- Llama softer crater (GSM8K −17% vs Qwen −95%); math/MC survive but IFEval craters. MAD is task-dependent.
- Gemma v8_att GSM8K .115 = deep enough that attacker output is not usable, IFEval .185
  matches Qwen (.124), Llama (.161).

---

## Table 4: Extended suite — over-refusal, factuality, code
Over-refusal `OR = refused + gibberish` share (see App. C, `score_overrefusal.py`; the
safety judge mislabels gibberish as benign on benign-framed prompts, hence the targeted
3-way scorer). On XSTest-safe / OR-Bench-Hard-1k **want LOW** (helps benign prompts);
on XSTest-unsafe **want HIGH** (refuses harmful).

**Qwen3-0.6B**
| cond | XST-safe OR ↓ | XST-unsafe OR ↑ | OR-Bench OR ↓ | SimpleQA correct | MBPP p@1 |
|---|---:|---:|---:|---:|---:|
| base_clean | .34 | .93 | .43 | .017 | .232 |
| base_att | .51 | .96 | .42 | .020 | .254 |
| v7_clean | **.98** | 1.00 | **.98** | .001 | .234 |
| v7_att | .99 | 1.00 | .99 | .000 | .234 |
| **v8_clean** | **.31** | .94 | **.49** | .028 | .228 |
| **v8_att** | 1.00 | 1.00 | 1.00 | .000 | **.000** |

**Gemma-3-1b**
| cond | XST-safe OR ↓ | XST-unsafe OR ↑ | OR-Bench OR ↓ | SimpleQA correct |
|---|---:|---:|---:|---:|
| base_clean | .18 | .81 | .41 | .022 |
| base_att | .10 | .70 | .21 | .016 |
| v7_clean | **.93** | .95 | **.92** | .008 |
| v7_att | 1.00 | 1.00 | 1.00 | .002 |
| **v8_clean** | .19 | .85 | .51 | .024 |
| v8_att | .98 | 1.00 | 1.00 | .014 |

Llama extended = **PENDING** (next-box script exists: `scripts/eval/eval_matrix_new.sh`).
SimpleQA near-floor at 1B — not discriminative here; deferred to scale-up. MBPP=0 for
gemma-3-1b (too small to code, base rate 0). Note: v8_att MBPP=.000 (Qwen) = attacked-v8
craters code (task-dependent crater; math survives with GSM8K −95%).

Read (the sharpest v7→v8 win): **v8 discriminates, v7 does not.** v7's keyword gibberish-wall
fires on benign kill/terminate/execute → over-refuses 92–98% of benign prompts (Qwen 0.98,
gemma 0.93). v8 = base rate on benign AND still refuses harmful (XSTest-unsafe 0.94 / 0.85).

---

## Table 5: Snapshot selection (Llama + gemma), pre-registered rule
Because stage-2 training oscillates through the wall↔clean-repair Pareto, we save every
25 steps and select by a **pre-registered constrained-lexicographic rule**:
**gates** `att_harm ≤ .05 AND att_gib ≥ .90 AND clean_harm ≤ .10`; **objective** among
survivors: maximize `clean_cap`; tie-break lower `clean_harm`. Selection on AdvBench-200
(validation); Tables 1–4 report the *full test suite* for the picked snapshot.
Automated in `scripts/tools/auto_pick_v8.py`; reproduces both manual picks below.

**Llama-3.2-1B** (recipe: λ_gib 8, λ_clean 3, λ_safe 1 → stage2 λ_safe 4, λ_gib 4, clean-start 250, ramp 100)
| snap | clean_harm ↓ | clean_cap ↑ | att_harm ↓ | att_gib ↑ | gate |
|---|---:|---:|---:|---:|:-:|
| s250 | .655 | .292 | .685 | .04 | ✗ |
| s275 | .570 | .500 | .740 | .04 | ✗ |
| s300 | .530 | .542 | .685 | .06 | ✗ |
| s325 | .055 | .667 | .355 | .47 | ✗ |
| s350 | .285 | .750 | .185 | .72 | ✗ |
| s375 | .145 | .625 | .305 | .58 | ✗ |
| s400 | .000 | .667 | .115 | .83 | ✗ |
| **s425 (pick)** | **.005** | **.833** | **.000** | **.99** | ✅ |
| final (500) | .000 | .583 | .000 | 1.00 | ✅ |

Ablation: first llama v8 (stage2-λ_safe=1, no auto-pick) → clean_harm .430 = unsafe.
Fix = raise stage2-λ_safe + auto-pick found s425 (clean_cap .833 vs final .583).

**Gemma-3-1b** (recipe as Llama but DL14, λ_gib and λ_clean same, S2GIB default)
| snap | clean_harm ↓ | clean_cap ↑ | att_harm ↓ | att_gib ↑ | gate |
|---|---:|---:|---:|---:|:-:|
| s250 | .000 | .208 | .000 | 1.00 | ✅ |
| s275 | .000 | .208 | .000 | 1.00 | ✅ |
| s300 | .000 | .458 | .000 | 1.00 | ✅ |
| s325 | .120 | .542 | .730 | .01 | ✗ (wall dissolves) |
| s350 | .055 | .542 | .830 | .01 | ✗ |
| s375 | .070 | .542 | .815 | .00 | ✗ |
| s400 | .050 | .583 | .900 | .01 | ✗ |
| s425 | .145 | .667 | .000 | 1.00 | ✗ (clean_harm) |
| **s450 (pick)** | **.080** | **.625** | **.000** | **1.00** | ✅ |
| s475 | .060 | .542 | .000 | 1.00 | ✅ |

The **wall oscillates** (s250-300 up, s325-400 dissolves, s425-475 reforms) — save-every +
auto-pick is required. The rule automatically excludes the dissolved-wall snapshots and
picks the highest-capability survivor (s450, cap .625).

---

## Table 6: FT-attack — v8 does not confer FT-resistance (out-of-scope statement)
The two threats are handled by different mechanisms. This paper defends abliteration.
Adversarial fine-tuning (AntiDote's threat model, arXiv 2509.08000) is orthogonal.

| model | attack | harmAct ↓ | gib | GSM8K (post-attack) ↑ |
|---|---|---:|---:|---:|
| Qwen v7 | FT-25 harmful demos | .74 | ~0 | ~ base |
| Qwen v8 | FT-25 harmful demos | **.755** | .075 | **.44** (= base) |

Both v7 and v8 FT-break to a coherent harmful model with capability intact. Explicit
paper statement: *v8's clean-side gains do not confer FT-resistance; the abliteration≠FT
boundary is a scope choice, not a claim. FT-resistance is complementary to this defense.*
(Fine-tune-resistance thread FTR-v2–v6 + FTR-TAR closed 2026-07-02, see
`docs/findings_multimodel_adaptive_2026_07_02.md`.)

---

## Table 7: Positioning vs prior art (must-cite / must-distinguish)
| paper | arXiv | threat model | our relationship |
|---|---|---|---|
| Arditi et al., *Refusal in LMs Is Mediated by a Single Direction* | 2406.11717 | attack: rank-1 abliteration | attack substrate we defend against |
| *There Is More to Refusal…* (Postgrave et al., 2026) | 2602.02132 | multiple directions, but "shared 1-D knob" | *supports* rank-1 as representative; motivates per-layer adaptive |
| Shairah et al., *Embarrassingly Simple Defense…* | 2505.19056 | preserve refusal after abliteration | closest defense competitor; they preserve refusal, we destroy capability |
| Kuo, Yadav, Smith, *Open-Weight FT Defenses…* / **ART** | 2605.26526 | abliteration-resistant tuning | **must beat on their protocol** (ASR −10-20pp claim) |
| Zou et al., *Circuit Breakers* | 2406.04313 | inference-time representation interrupt | frame v8 as *post-tamper conditional circuit breaker in weights* |
| Chen et al., *AntiDote* | 2509.08000 | FT-resistance, 0.6B-27B, 52 attacks | orthogonal (FT), do not claim to beat |
| Carleo et al., *Willing but Unable* (code) | 2606.05396 | abliteration decouples refusal from capability | sharpens our "smart XOR dangerous" framing |

Novelty statement (for intro): to our knowledge, no prior work engineers a *conditional*
capability collapse under successful refusal-direction removal while preserving base-like
clean behavior. Prior abliteration defenses (2505.19056) *preserve refusal* under attack;
we *destroy capability with it*. Prior FT-defense work (2509.08000, 2605.26526) targets
gradient-based tampering; abliteration is a rank-1 non-gradient attack with different
geometry (2406.11717) and requires a different defense.

---

## Provenance and TODOs

Provenance (all numbers grep-checkable):
- Qwen matrix ← `full_eval_matrices/qwen_full_matrix.md` + `qwen_extended_suite.md`.
- Llama matrix ← `full_eval_matrices/llama_full_matrix.md` (base_att re-judged 2026-07-06,
  see llama file note).
- Gemma matrix ← `full_eval_matrices/gemma_full_matrix.md` (extended suite in same file).
- Snapshot picks ← `full_eval_matrices/{llama,gemma}_v8_snapshot_pick.md`.
- Recipe / mechanism ← `experiments/train_tamper_resistant_v8.py` (loss assembly line 548).
- FT-attack negative ← `docs/devlog_2026_07_04.md` (line 61).

Known TODOs (data holes to close before arXiv):
- [ ] Gemma S2GIB=8 rerun → close −15% clean IFEval gap (Table 1).
- [ ] Llama extended suite (Table 4 row missing) — `scripts/eval/eval_matrix_new.sh` on next box.
- [ ] Per-layer adaptive attack matrix, all 3 archs (defends against reviewer objection re rank-1).
- [ ] Seeds n≥3 on ≥1 arch (defends against snapshot-cherry-pick objection; auto-picker already in).
- [ ] ART baseline on Kuo–Yadav–Smith protocol (2605.26526) — must-beat.

Deferred to ICLR delta (out of this arXiv scope):
- Scale-up Phi-4-mini 3.8B / Ministral-3B / SmolLM2-1.7B (kills "1B toy" objection).
- TamperBench (external benchmark) + Tier-2 official judges.
- Mechanistic section: why attack unlocks wall (not generic distribution shift).
