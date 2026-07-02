# CLAUDE.md — read first

**Current state lives in `docs/handoff_2026_07_02_v2.md`. Read that, then `MEMORY.md`,
then this file for durable conventions.** Older handoff text below the line is
historical (P0/P1 era) — do not act on it.

## Project in one paragraph
tamperforge = a pre-release procedure that entangles safety with capability in
open-weight LLMs so cheap uncensoring self-defeats ("smart-and-safe XOR
dumb-and-dangerous"). Two threads: **ABL** (abliteration-resistance — the strong,
working result) and **FTR** (fine-tune-resistance — iterating, not yet won).
Model: `google/gemma-3-1b-it`.

## Naming (USE THIS — two version lines collided on v5/v6/v7)
- **ABL-v{n}** = abliteration line, `outputs/tamper_resistant_p1b_v{n}.pt`. ABL-v7 =
  product. Materialized dirs: `outputs/abl_v7_hf`, `outputs/abl_v7_hf_attacked`.
- **FTR-v{n}** = fine-tune-resistance line, `outputs/ft_resistant_p4_v{n}.pt`.

## Headline results (as of 2026-07-02)
- **ABL-v7 works AND generalizes:** abliterating it -> 100% gibberish / 0% ASR across
  prefill + HarmBench + BeaverTails (off-distribution, non-gradient attacks it was
  never trained on); abliterating base -> real harm (ASR 0.41-0.80). Full table:
  `docs/findings_prefill_harmbench_beavertails_2026_07_02.md`.
- **Caveat:** clean ABL-v7 has real off-AdvBench coherence cost (13.5-50% gibberish,
  no attack). "Free product" only held on AdvBench. Being measured now.
- **FTR: no win.** v2-v5 = 1-shot moat (artifact), v6 = LoRA-inner TAR, meta-lr sweep
  inconclusive. Full K=0..200 validation sweep is the pending verdict.

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
- Judge locally via conda: `~/miniforge3/envs/env_ml/bin/python` (NOT a venv path).

## Infra
- **4090** `vast_tamperforge` (ssh9.vast.ai:33059) `/venv/main`, torch 2.11+cu130,
  vLLM 0.24 — WORKS. Eval + capability box.
- **5090x2** `tamperforge_5090x2` (ssh5.vast.ai:24813) torch 2.12+cu130, 32GB x2 —
  vLLM was broken (NCCL symbol mismatch); verify `import torch,vllm` before use.
  FTR training/validation box (`CUDA_VISIBLE_DEVICES=0/1` = two independent jobs;
  code is single-GPU per job).
- Box git = private HTTPS, needs a PAT in the remote to push. scp code / pull results
  to local + push from there if box git is uncooperative.
- Backups: GitHub (code/docs/results), private HF `aaronrockmenezes/tamperforge`
  (.pt + model dirs, `scripts/push_to_hf.py`).

## Key scripts
- `experiments/train_ft_resistant_v6.py` — FTR Lever-2 (LoRA inner + judge gate + FO-MAML).
- `experiments/ft_attack.py` — the FT attack (validation) + `--n-shots K` sweep.
- `experiments/prefill_attack.py`, `p0_baseline_eval.py --prompt-source {advbench,harmbench,beavertails}`.
- `experiments/judge_generations.py`, `experiments/save_p1b_checkpoint.py` (materialize .pt [+attack]).

---
# HISTORICAL (P0/P1 era — do not act on)

The original P0/P1 handoff text is preserved in git history and `HANDOFF.md`. It
predates the ABL/FTR split, the abliteration battery, and the FT work. Ignore its
"immediate next steps."
