# tamperforge — session memory / handoff index

> Repo-local state index. Updated 2026-07-18. Read `docs/devlog_2026_07_17.md` first (newest,
> most important: Heretic breaks ABL-v8 on Llama), then `docs/handoff_2026_07_03_MASTER.md`
> (multi-model + attack-robustness campaign) for the full prior state, `CLAUDE.md` for conventions.

## Thread 2 — Scale attempt + third-party validation + adaptive-attack crack (2026-07-17/18, `docs/devlog_2026_07_17.md`)
- v8 = 3/3 architectures proven as of 2026-07-04 (gemma/Qwen/Llama), see `devlog_2026_07_04.md`.
- **Qwen3-8B thinking-mode scale attempt: parked, not closed.** Trained successfully after 3
  OOM iterations (`TRAIN_SCOPE=last_half`+`adamw8bit`+`empty_cache`); wall dissolve/reform
  confirmed at scale. Pick-job's raw generations destroyed by a `RunLogger` reuse bug in
  `auto_pick_v8.py` (fixed, commit `d89872d`) — s400/425/500 snapshots pushed to HF instead of
  re-running the full pick. Resuming needs a fresh pick job + four-cell eval.
- **TamperBench (third-party benchmark, arXiv 2602.06911) validation: in progress, not
  complete.** 3 undocumented bugs in TamperBench's own code fixed (fp64 hidden upcast +
  zombie CUDA context, see `common_issues.md`); only `gemma_base` run confirmed clean so far,
  full 6-run sweep (3 archs × base/v8) not finished.
- **HERETIC BREAKS ABL-v8 ON LLAMA-3.2-1B.** (`docs/heretic_v8_llama_2026_07_18.md`) Adaptive
  KL-optimizing abliteration gets 40% coherent judged harm at ZERO capability cost at
  mid-strength, 88% harm at its strongest trial (only IFEval shows any cost even then, GSM8K
  stays clean-level). Not gemma v7's graceful gibberish-leak pattern under Heretic — a clean,
  low-cost, controllable harm/refusal trade. **Do not claim "survives Heretic" as blanket** —
  true for gemma v7, false for Llama v8. Not yet tested: gemma v8/Qwen v8 vs Heretic (the
  open question — architecture-specific or product-specific crack?).

## Naming (version lines; DON'T conflate)
- **ABL-v{n}** = abliteration line, `outputs/tamper_resistant_p1b_v{n}.pt`. ABL-v7 =
  product. Dirs: `outputs/abl_v7_hf` (clean), `outputs/abl_v7_hf_attacked` (attacked).
- **FTR-v{n}** = fine-tune-resistance line, `outputs/ft_resistant_p4_v{n}.pt` (v2–v6).
- **FTR-TAR** = faithful-TAR FT-resistance attempt (successor to FTR-v6; dropped "v7" to
  avoid ABL-v7 collision). `train_ft_resistant_tar.py`, stem `ft_resistant_p4_tar`.

## Thread 1b — Multi-model + ADAPTIVE attack (2026-07-02, `docs/findings_multimodel_adaptive_2026_07_02.md`)
- **Generalizes to 3 architectures** (per-model refusal layer): gemma FULL WIN (gibberish +
  capability crater, reproduced seed42+seed2); Qwen3-0.6b CLEAN DEFENSE (gibberish wall, no
  crater); Llama-3.2-1b@L13 PARTIAL (leaky ~13%, no crater; diffuse safety).
- **FLAGSHIP: survives a PER-LAYER ADAPTIVE attack.** `save_p1b_checkpoint --per-layer` (indep
  diffmeans dir per layer, our infra, no AGPL). OG gemma: base → 0.712 coherent harm / v7 →
  **0.000 harm, 100% gibberish, ARC 0.34→0.27 MMLU 0.39→0.24**. No better than rank-1 → robust
  to adaptive abliteration, not just the cheap attack. Qwen/Llama per-layer attacks running.
- **DL bug:** codex's 50%-depth guess was WRONG for Llama (real refusal L13, not L8) — cost
  harmAct 0.65→0.13 when fixed. Per-model layer selection essential. Full sweeps = HIGH-PRI TODO.
- **keyword ASR unreliable BOTH ways** (Qwen v7 kw0.99/judge0.01; Llama base kw0.89/judge0.60).
- **seed fragility** ~half seeds fail (fp32 gib-CE fix killed NaN crashes but not the basin
  issue); n=2 gemma final.

## Thread 1 — Abliteration (ABL-v7): WORKS + generalizes + craters attacker capability
FULL-DATA 2x2x3 judge battery + lm_eval capability, done 2026-07-02
(`docs/findings_prefill_harmbench_beavertails_2026_07_02.md`):
- **Generalizes off-distribution.** Abliterating ABL-v7 -> ~0% ASR / ~100% gibberish
  across prefill(520)/HarmBench(200)/BeaverTails(1483), incl. prefill = non-gradient
  attack never trained on. Abliterating base -> real harm ASR 0.39-0.73.
- **STRONGEST result: abliteration craters CAPABILITY too.** Abliterating ABL-v7:
  ARC 0.352->0.265 (-25% rel), MMLU 0.395->0.268 (-32% rel). Abliterating base barely
  dents it (ARC 0.352->0.358, MMLU 0.395->0.379). MAD proven on a benchmark, not just
  gibberish-on-harmful. Attacker gets a measurably dumber model.
- **Clean product credible on MC:** clean ABL-v7 ~= base on ARC 0.344 / MMLU 0.393.
  Gibberish concentrated on harmful-prompt distributions, not ARC/MMLU.
- **GSM8K (2026-07-02):** abliterating ABL-v7 -> 0.006 (~zero, −98% vs base 0.256) =
  capability-collapse corroborated on GENERATIVE math, even more total than MC. BUT
  clean ABL-v7 GSM8K 0.167 (−35% vs base) = "free product" holds on MC, NOT on
  chain-of-thought math. New honest limitation (with the prefill hole).
- **Honest limitation:** CLEAN ABL-v7 has a prefill hole (ASR 0.323, harmAct 0.285 >
  base 0.108) + off-AdvBench clean gibberish (11/17/47.5% prefill/HB/BT). Report in
  Limitations.

## Thread 2 — Fine-tune resistance (FTR): NO WIN (v6 = lobotomy, confirmed)
- FTR-v2..v5 = 1-shot moat at best (artifact); all -> base by K>=5. Crux: inner-sim
  defended teacher-forced CE, not generation.
- **FTR-v6 = Lever-2 (LoRA-inner TAR), FAILED.** lr2e4 looked promising on a
  n=100/128tok subset (ASR~0 to K50) but the GATE killed it: clean ARC **0.217** /
  MMLU **0.246** ~= chance (0.25) -> model is broken at the core (MC loglikelihood, so
  not a generation artifact). Full-520/512 attacked sweep confirms: harmAct **0.000**
  at every K (gibberish everywhere). The "moat" was gibberish-in/gibberish-out on an
  already-dead model, NOT the MAD signature. The K100 harmAct 0.21 in the subset was a
  128-tok short-output artifact. lr5e5 also dead. **DISCARD both v6 ckpts.**
- **Lesson (methodological win for paper):** appearing FT-resistant via capability
  collapse is indistinguishable from real resistance WITHOUT a capability eval;
  ASR-alone called this a win. `usefulness_label` + MC-capability check caught it.
- **FTR-TAR = FAILED (2026-07-02). FT THREAD CLOSED.** `train_ft_resistant_tar.py`,
  faithful TAR (KL-to-ABL-v7 retain anchor + bounded TR). 2 runs (λ_retain 4 & 8),
  killed ~step 75/200: **L_tr pinned at ceiling ~7.99 the whole time = θ cannot reduce
  the attack's success at all** (frac_comply flat ~0.9). retain_KL climbing (0.5→0.8
  @ret4, 0.5→1.3 @ret8) = drifting, not stabilizing. Preserves capability better than
  v6's lobotomy but STILL can't out-harden a rank-16 LoRA FT attack at meta-lr 1e-5.
  STOP RULE hit (≫5 runs across v2-v6+TAR). **Decision: anchor paper on abliteration;
  FT = honest characterized-cost negative.**

## FT frontier — harmful_actionable at FULL 520/512 (judged, coherent harm)
| K | base OG | ABL-v7 | FTR-v6 (LOBOTOMY) |
|--:|--:|--:|--:|
| 0 | 0.012 | 0.004 | 0.000 (gib 0.998) |
| 25 | 0.833 | 0.869 | 0.000 |
| 50 | 0.815 | 0.833 | 0.000 |
| 100 | 0.827 | 0.798 | 0.000 |
| 200 | 0.763 | 0.837 | 0.000 |
- **base ≈ ABL-v7:** both break by K=25 to ~0.8 harmAct, COHERENT (gibberish ~0).
  ABL-v7 gives ZERO FT-resistance (= base). Abliteration-resist ≠ FT-resist, confirmed.
  Attack saturates fast (flat K25→200); real knee is K=1–25.
- **FTR-v6 harmAct 0.000 is FAKE** = lobotomy (gibberish everywhere, ARC 0.217/MMLU 0.246
  ~chance). harmAct-alone ranks the broken model as best defense -> METHODOLOGY FIGURE.
- **FTR-TAR must beat this band:** low harmAct at high K AND coherent/capable (ARC/MMLU
  ~= ABL-v7). Nothing has done both yet.

## Eval tooling (built this session)
- `prefill_attack.py` (compliant-prefix forcing), `p0_baseline_eval.py --prompt-source
  {advbench,harmbench,beavertails}`, `--n-prompts` (default -1 = FULL).
- `judge.py`: `coherent` 0/1 + `usefulness_label()` (refused/gibberish/harmful_actionable/
  harmful_vague/benign). Summary: usefulness_counts + harmful_actionable_rate + gibberish_rate.
- `load_harmbench` (walledai, gated-access-ok), `load_beavertails`.

## Paper framing (from critiques + findings)
Anchor: **abliteration-resistance + MAD mechanism, generalizing off-distribution**;
FT = characterized cost-frontier (not solved); honest clean-coherence-cost limitation.
Title dir: "Cheap Abliteration of Open-Weight LLM Safeguards Can Be Made
Capability-Destructive." Full plan: `docs/archive/critiques.md`, `docs/archive/next_steps_2026_07_02.md`.

## Infra (details in CLAUDE.md / handoff_v2)
- 4090 `vast_tamperforge` (vLLM works) = eval box. 5090x2 `tamperforge_5090x2`
  (vLLM was NCCL-broken; verify before use) = FTR box, CUDA_VISIBLE_DEVICES=0/1.
- Judge local via `~/miniforge3/envs/env_ml/bin/python`, 64 workers, 512 max-tokens.
- Never nohup box cmds. Full datasets only. GitHub + private HF backups.
