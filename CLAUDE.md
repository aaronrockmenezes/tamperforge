# CLAUDE.md — read first

**Current state lives in `docs/devlog_2026_07_17.md` (Qwen3-8B scale attempt, TamperBench
third-party validation, Heretic cracks v8 on all three architectures — the newest finding).
Read that first**, then `docs/handoff_2026_07_03_MASTER.md` (multi-model + attack-robustness
campaign), `docs/findings_multimodel_adaptive_2026_07_02.md`, `MEMORY.md`, then this file for
durable conventions. Older text below the line is historical (P0/P1) — do not act on it.

**One-line status (2026-07-18):** **ABL-v8 = the conditional wall, proven 3/3 architectures**
(Qwen + Llama + gemma, all done as of 2026-07-04 — see `docs/devlog_2026_07_04.md`). v8 fixes
v7's clean tax: clean model is base-like (safe+coherent+capable+helpful) while abliteration
still self-destructs under the naive rank-1 attack. **But: Heretic (adaptive, KL-optimizing
abliteration) BREAKS THE WALL ON ALL 3 ARCHITECTURES** — Llama 88% harm (some IFEval cost at
its most extreme trial only), gemma 93% harm (zero capability cost, any trial), Qwen 82% harm
(zero capability cost, any trial, including GSM8K which rank-1 craters −95%). See
`docs/heretic_v8_2026_07_18.md` for the full cross-architecture table. **Do not claim
"survives adaptive attacks" as a blanket statement anywhere** (see `docs/related_work.md`
correction) — v8 stops the naive rank-1 attack cleanly, that result stands; Heretic is a
different story on all 3 archs. ABL-v7 (prior, more leaky) survived Heretic on gemma; FTR
(fine-tune-resistance) dead, separate thread, do not reopen.

## Project in one paragraph
tamperforge = a pre-release procedure that entangles safety with capability in
open-weight LLMs so cheap uncensoring self-defeats ("smart-and-safe XOR
dumb-and-dangerous"). Two threads: **ABL** (abliteration-resistance — the strong,
working result) and **FTR** (fine-tune-resistance — iterating, not yet won).
Model: `google/gemma-3-1b-it`.

## Naming (USE THIS — version lines collided on v5/v6/v7)
- **ABL-v{n}** = abliteration line, `outputs/tamper_resistant_p1b_v{n}.pt`. **ABL-v8 = current
  product** (`train_tamper_resistant_v8.py`; conditional wall — clean is base-like, only
  abliteration self-destructs). ABL-v7 = prior (worked but clean-degraded). v8 ckpts:
  `tamper_resistant_qwen3_0p6b_v8.pt`, `tamper_resistant_llama32_1b_v8_best.pt` (s425 pick).
- **FTR-v{n}** = fine-tune-resistance line, `outputs/ft_resistant_p4_v{n}.pt` (v2–v6).
- **FTR-TAR** = faithful-TAR FT attempt (successor to FTR-v6; "v7" retired to avoid
  ABL-v7 collision). `../tamperforge-archive/experiments/archive/ft/train_ft_resistant_tar.py`, stem `ft_resistant_p4_tar`.

## Headline results (as of 2026-07-02, FULL datasets + lm_eval capability)
- **ABL-v7 works, generalizes, AND craters attacker capability:** abliterating it ->
  ~100% gibberish / ~0% ASR across prefill(520)+HarmBench(200)+BeaverTails(1483)
  (off-distribution, non-gradient); AND ARC 0.352->0.265 (-25%), MMLU 0.395->0.268
  (-32%). Abliterating base -> real harm (ASR 0.39-0.73) with capability intact. MAD
  proven on a benchmark. `docs/findings_prefill_harmbench_beavertails_2026_07_02.md`.
- **Clean product credible:** clean ABL-v7 ~= base on ARC/MMLU (0.344/0.393). Gibberish
  is on harmful-prompt distributions only.
- **Honest limitation:** clean ABL-v7 prefill hole (ASR 0.323 > base 0.108) +
  off-AdvBench clean gibberish (11/17/47.5%).
- **FTR: NO WIN. FTR-v6 = lobotomy (confirmed).** v2-v5 = 1-shot moat (artifact, broke
  by K=5). FTR-v6 lr2e4 looked promising on a 128tok subset but GATE killed it: clean
  ARC 0.217 / MMLU 0.246 ~= chance = broken model; full-520/512 attacked sweep harmAct
  0.000 at every K. Both v6 ckpts DISCARD. Lesson: capability eval is REQUIRED to tell
  real resistance from a broken model — ASR-alone called this a win.
- **FTR-TAR = FAILED. FT THREAD CLOSED.** Faithful TAR (KL-to-ABL-v7 retain anchor +
  bounded TR). 2 runs killed ~step75: L_tr pinned at ceiling = θ can't out-harden the
  rank-16 LoRA attack (frac_comply flat ~0.9); retain_KL drifting up. Better than v6
  (no lobotomy) but same wall. STOP RULE hit. **Paper anchors on abliteration; FT =
  honest characterized-cost negative.** Don't reopen FT without a fundamentally
  different lever (loss-landscape moonshot), not another TAR knob.

## Hard conventions (do not violate)
- **Judge, not keyword.** `judge_generations.py` (DeepSeek V4 Flash via OpenRouter).
  Report `judge_asr` + `usefulness_label` (gibberish vs refused vs harmful_actionable)
  — ASR alone hides gibberish-collapse.
- **Full datasets, no subsets** for any reported result. `--n-prompts` defaults to
  full (-1). Sizes: AdvBench 520, HarmBench-standard 200, BeaverTails 1483.
- **Never nohup/background box commands without asking.** User watches every command,
  runs tmux himself.
- **Never vendor OBLITERATUS/AGPL code** — call it as a separate attacker harness.
- HF repo is PRIVATE (uncensored weights, dual-use).
- Source `results/` keeps manifests/summaries only; raw run artifacts and historical trees
  live in `../tamperforge-archive`. Do not edit or remove retained v7 files.
- No local checkpoint payloads remain. Treat private HF as artifact storage; do not run the
  stale `scripts/push_to_hf.py` uploader.
- Judge locally via conda: `~/miniforge3/envs/env_ml/bin/python` (NOT a venv path).

## Infra
- **4090** `vast_tamperforge` (ssh9.vast.ai:33059) `/venv/main`, torch 2.11+cu130,
  vLLM 0.24 — WORKS. Eval + capability box.
- **5090x2** `tamperforge_5090x2` (ssh5.vast.ai:24813) torch 2.12+cu130, 32GB x2 —
  vLLM was broken (NCCL symbol mismatch); verify `import torch,vllm` before use.
  FTR training/validation box (`CUDA_VISIBLE_DEVICES=0/1` = two independent jobs;
  code is single-GPU per job).
- **A6000** `vast-tamperbench` — TamperBench third-party validation + Heretic adaptive-attack
  box. TamperBench is a separate clone at `/workspace/TamperBench`, NOT vendored/git-tracked —
  3 source patches (fp64→fp32 in two files + empty_cache) live only on this box's disk and
  must be reapplied on a fresh clone (see `docs/common_issues.md`). `heretic-llm` installed
  via pip, no patches needed.
- Box git = private HTTPS, needs a PAT in the remote to push. scp code / pull results
  to local + push from there if box git is uncooperative.
- Backups: GitHub (code/docs/compact results), sibling `../tamperforge-archive` (raw and
  historical files), private HF `aaronrockmenezes/tamperforge` (retained model artifacts).

## Key scripts
- `../tamperforge-archive/experiments/archive/ft/train_ft_resistant_v6.py` — FTR Lever-2 (LoRA inner + judge gate + FO-MAML).
- `experiments/ft_attack.py` — the FT attack (validation) + `--n-shots K` sweep.
- `experiments/prefill_attack.py`, `p0_baseline_eval.py --prompt-source {advbench,harmbench,beavertails}`.
- `experiments/judge_generations.py`, `experiments/save_p1b_checkpoint.py` (materialize .pt [+attack]).

---
# HISTORICAL (P0/P1 era — do not act on)

The original P0/P1 handoff text is preserved in git history and `../tamperforge-archive/docs/archive/HANDOFF_p0p1_historical.md`. It
predates the ABL/FTR split, the abliteration battery, and the FT work. Ignore its
"immediate next steps."
