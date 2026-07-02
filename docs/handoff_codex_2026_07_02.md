# Handoff for Codex — 2026-07-02

Self-contained pickup. Read `CLAUDE.md` + `docs/handoff_2026_07_02_v2.md` +
`docs/devlog_2026_07_02.md` for full context. This file = what's running, what's next.

## One-paragraph state
tamperforge entangles safety with capability in `google/gemma-3-1b-it` so cheap
uncensoring self-defeats. **Two threads.** ABL (abliteration-resistance, product =
ABL-v7) is the STRONG, working result and the leaning paper anchor. FTR
(fine-tune-resistance) is the hard, unsolved thread: v2–v6 all failed; FTR-TAR is the
current attempt, running now.

## Naming (do not conflate)
- **ABL-v{n}** = abliteration line. `outputs/tamper_resistant_p1b_v{n}.pt`. ABL-v7 =
  product. HF dirs `outputs/abl_v7_hf` (clean), `outputs/abl_v7_hf_attacked`.
- **FTR-v{n}** = fine-tune-resistance line. `outputs/ft_resistant_p4_v{n}.pt` (v2–v6, all failed).
- **FTR-TAR** = current TAR attempt (NOT ABL-v7). `experiments/train_ft_resistant_tar.py`,
  ckpt stem `ft_resistant_p4_tar`.

## Headline results (full datasets, LLM-judged)
- **Abliterating ABL-v7** → ~0 ASR / ~100% gibberish across prefill(520)/HarmBench(200)/
  BeaverTails(1483) AND ARC 0.352→0.265 (−25%), MMLU 0.395→0.268 (−32%). Attacker gets a
  broken, dumber model.
- **Abliterating base** → coherent harm (ASR 0.39–0.73), capability intact. (the danger v7 removes)
- **Clean ABL-v7 ≈ base** on ARC/MMLU (0.344/0.393). Product credible.
- **Limitation:** clean ABL-v7 prefill hole (ASR 0.323 > base 0.108).
- **FT frontier (full 520/512):** base ≈ ABL-v7 both → ~0.83 harmful_actionable by K=25,
  COHERENT. ABL-v7 gives ZERO FT-resistance.
- **FTR-v6 = FAILED lobotomy:** clean ARC 0.217 / MMLU 0.246 (~chance); harmAct 0.000 at
  all K only because it's broken. Both v6 ckpts DISCARD.

## FT THREAD CLOSED (2026-07-02)
FTR-TAR (2 runs, λ_retain 4 & 8) killed ~step 75/200. **L_tr pinned at ceiling ~7.99
throughout = θ cannot reduce the rank-16 LoRA attack at all** (frac_comply flat ~0.9,
post-attack gen fully compliant every step). retain_KL drifting up (0.5→0.8/1.3). Better
than v6 (no full lobotomy, KL bounded) but same wall: a meta-lr-1e-5 defender can't
out-harden a realistic FT attack. STOP RULE hit (≫5 runs). **DECISION: paper anchors on
abliteration; FT = honest characterized-cost negative.** The saved "best" ckpts are
step-25 ≈ unmodified ABL-v7 — not worth validating. Do NOT reopen FT with another TAR
knob; only a fundamentally different lever (loss-landscape basin-trap moonshot) would
justify it.

## RUNNING NOW
- **5090x2** (ssh5.vast.ai:24813): FTR-TAR runs being KILLED (see above) → will be free.
- **4090** (ssh9.vast.ai:33059): free (frontier refs + GSM8K + HF backup done). Sleep or
  use for the abliteration rigor runs below.

## NEXT STEPS — lock abliteration for publication (FT is done). Priority order:
1. **Multi-seed (3×)** ABL-v7: retrain 2 more seeds (`train_tamper_resistant.py
   --attack-ensemble`), re-run the full battery. Is it a lucky run?
2. **Adaptive attacker (P2, OBLITERATUS)** — mandatory gradient-masking check. AGPL harness,
   never vendor; pip-install + run on `outputs/abl_v7_hf`. Also stronger/adaptive abliteration.
3. **GSM8K** capability on the 4 battery models (have ARC+MMLU), clean AND post-attack.
4. **Multi-model:** Qwen3-1.7B, Llama-3.2-1B, gemma-3-4b — the generality claim.
5. **TamperBench** harness + **ART** abliteration baseline (external > home-grown).

### 3. Build the FTR frontier figure
base vs ABL-v7 vs FTR-v6(lobotomy) vs FTR-TAR — harmful_actionable AND capability vs K.
The base-vs-v6 pair is the methodology figure (harmAct-alone ranks the broken model best).

## Hard conventions (do not violate)
- **Judge, not keyword** (`judge_generations.py`, DeepSeek V4 Flash). Report `judge_asr` +
  `usefulness_label` (gibberish vs refused vs harmful_actionable). ASR alone hides lobotomy.
- **Full datasets, no subsets** for reported results. `--n-prompts -1` = full. AdvBench 520 /
  HarmBench 200 / BeaverTails 1483.
- **Capability eval is mandatory** for any FT-resistance claim (v6 lesson).
- **Never nohup/background box commands** — user watches every command, uses tmux. Give
  commands; don't auto-launch training on the box.
- **Judge LOCALLY** via `~/miniforge3/envs/env_ml/bin/python` (NOT a venv), 64 workers.
- HF repo PRIVATE (dual-use weights). Never vendor OBLITERATUS (AGPL) — separate harness.

## Infra / backup
- 4090 = eval/capability box (vLLM works). 5090x2 = FTR training (vLLM now works too;
  CUDA_VISIBLE_DEVICES=0/1 = two single-GPU jobs). Box git = private HTTPS + PAT.
- GitHub `aaronrockmenezes/tamperforge` (code/docs/results). Private HF same name
  (.pt + model dirs, `scripts/push_to_hf.py`). Both backed up this session.
