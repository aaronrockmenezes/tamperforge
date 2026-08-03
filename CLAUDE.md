# CLAUDE.md — read first

**Current state: `docs/handoff_2026_08_03_codex.md`** (2026-08-03, most recent), then
`docs/handoff_2026_08_02_llama_and_mechanism.md`, then
`docs/handoff_2026_08_01_version_a_b.md` (Qwen results table).

**FOUR CORRECTIONS FROM 2026-08-03 that override text below — read before acting on anything.**

1. **The judge was never actually pinned** despite the claim. All five judge entry points
   defaulted to the floating `deepseek-v4-flash`; they now default to `deepseek-v4-flash-0731`.
   **Every MT-Bench number predating 2026-08-03 came from the floating tag — do not compare
   across the two, re-judge.** Base is **4.54**, gate-1 bar **4.04**.
2. **Never report an n=1 MT-Bench score.** Identical generations, same pinned judge, temp 0
   scored base 4.54, then 4.74. That moved the gate-1 bar and flipped a verdict.
   `mtbench_single.py --repeats` defaults to 3. The noise is WITHIN one judge, so changing tags
   does not fix it.
3. **ARC/MMLU/GSM8K are necessary, never sufficient — always pair with MT-Bench.** Heretic'd
   version_B holds 95% GSM8K at MT-Bench 2.94, below its own clean 3.30. **version_B's wall IS
   its fluency damage.** This instrument trap has now fired three times.
4. **The poison pill never fires under heretic** — 93–102% capability retention for every model
   and seed. The only crater anywhere is version_B under rank-1. Consistent with the read/write
   mechanism below.

**And a new attack: benign SFT.** 957 rows of ordinary assistant data, no harmful examples,
5 minutes, takes version_B's rank-1 harm 0.0000 → 0.3673 while *improving* MT-Bench 3.30 → 4.22.
The Tier-4 "FT is out of scope" exclusion does not cover it — that exclusion assumes FT attacks
need harmful demonstrations. See `docs/attack_zoo_v0.md`.

**INFRA: THERE IS NO BOX (2026-08-03).** `vast-versiona-3090` was destroyed after version_G
finished training; the 4090/5090x2/A6000 entries in the Infra section below never resolved
either. **Provision a new box and rsync the repo to it** — the box is never a git checkout, so
deploy by rsync and commit from local.

**Everything survived**: all checkpoints on private HF under `final_backup_2026_08_03/`, raw
logs/generations/metrics in `../tamperforge-archive/box_2026_08_03/`, code + summaries in git.
Clean models and attacked snapshots were NOT saved by design — they regenerate from the `.pt`.

**START HERE: `version_g_qwen_500.pt` is trained and completely unevaluated**, and its rerouting
loss did something no other arm has (`L_rr` 0.9967 → 0.1702). See `TODO.md`.

**THE MECHANISM (2026-08-02).** MAD fires on **read-projection** ablation. Heretic ablates
**write projections only** (`attn.o_proj`, `mlp.down_proj`) and therefore never triggers it —
at any direction layer, any alpha. Adding read projections to Heretic's own winning attack
drops harm 0.3231 -> 0.0577 (82%). version_C trained write-only attacks ~60% of steps and
still could not place the entanglement there. **This is structural, not a coverage gap: do
not build another attack sampler.** Sampling has now failed fixed (v8), widened (version_A/B)
and adaptive-against-a-live-optimiser (version_C).

**MAD IS QWEN-ONLY (2026-08-02).** version_B on Llama-3.2-1B blocks every attack (harm
0.000-0.065, gibberish 0.83-1.00) with capability INTACT across five benchmarks — ARC 101%,
GSM8K 99-103%, HumanEval 109%, MBPP 98-105%, MMLU-12 91-92% the only mover. That is a
**fortress** (the occupied Shairah/ART cell), not the poison pill this project claims. On Qwen
the same two cheap attacks take GSM8K to 2% and 27%. Never state MAD as a general result on
one architecture. Full matrix: `results/llama_no_approx_2026_08_02/summary.json`.
Also: version_B costs Llama real CLEAN capability, visible only in code (HumanEval 0.2866 vs
base 0.3476 = 82%) — ARC/MMLU/GSM8K all missed it.

**SHAIRAH BASELINE IS RUN (2026-08-02) AND IT BEATS US EXCEPT ON ONE AXIS.**
`results/shairah_2026_08_02/summary.json`. Extended-refusal SFT (one fine-tune, no adversarial
training) blocks rank-1 and surgical on BOTH architectures by headroom, keeps 86-104% of
capability, and its clean model beats every version of ours on usability (XSTest safe benign
0.624 Qwen / 0.860 Llama vs our best 0.436 / 0.316) and capability (Qwen GSM8K 0.4610 vs
version_B 0.3889). **Our only surviving differentiator: on QWEN, abliterating version_B takes
GSM8K to 2% (rank-1) / 27% (surgical) while Shairah stays at 91% / 86% — capability collapse,
the poison pill. On LLAMA Shairah strictly dominates us.** Narrow the claim accordingly.

**THE CLEAN MODEL OVER-REFUSES BADLY (XSTest, 2026-08-02).** version_B's CLEAN model answers
only 43.6% (Qwen) / 31.6% (Llama) of XSTest's 250 safe-but-scary prompts, vs base 74.4% /
81.2% — the rest split between over-refusal and outright GIBBERISH on benign input (Qwen
0.280, Llama 0.188). Both architectures, so it is the recipe, not MAD. AdvBench-only
evaluation hid this completely; **never call the clean product "base-like" on AdvBench
numbers alone.** Worse, heretic partially UNDOES the over-refusal (Llama attacked benign
0.572 > defended clean 0.316), so the attacker gets a more usable model. Safety does improve
(unsafe refusal 0.890/0.975 vs 0.780/0.915). `results/xstest_2026_08_02/summary.json`.

**MEASURE AGAINST THE BASE CEILING, not defended-clean** — that denominator error ran through
a whole day of analysis. Ceilings (judged, 520 walledai, capability intact):
Qwen 0.2577 -> 0.6596 · Llama 0.0019 -> 0.8269. Llama has ~2x the dynamic range and heretic is
its APEX attack.

**Version status.** version_A strongest vs heretic (~0% of headroom, n=5); version_B best
demonstrates the mechanism (rank-1 AND surgical self-destruct) and its collapse is
**KL-dependent, firing above ~0.12**; version_C is a regression — it pushed heretic's whole
front into the high-KL band and simultaneously lost the collapse that punishes it. **Attacker
*cost* is a dead claim — never repeat "must accept 5x the perturbation".**

**Framing is narrower than it looks.** SEAM (arXiv:2505.12186) already published the
poison-pill idea for FINE-TUNING and ART (arXiv:2605.26526) already published
abliteration-resistant training. Our unoccupied cell is abliteration x capability-collapse
only. `TODO.md` [HIGH]: the Shairah extended-refusal baseline is still unrun, and until it is
we cannot claim to beat a method that needs no adversarial training.

Then `docs/devlog_2026_07_17.md` (Qwen3-8B scale, TamperBench validation),
`docs/handoff_2026_07_03_MASTER.md`, `docs/findings_multimodel_adaptive_2026_07_02.md`,
`MEMORY.md`, then this file for durable conventions. Older text below the line is historical
(P0/P1) — do not act on it.

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
- **The vX scheme STOPS AT v11.** New defense work is **version_A**, version_B, … (files
  `version_a_*.py`, run ids `version_a_*`). Nothing is renamed retroactively: ABL-v8/v9/v10
  and the v11 probe scripts keep their existing meanings. version_A = the first line to
  train against capability-overlap as an attack axis (`experiments/version_a_attack.py`);
  version_B adds heretic's own axes (per-projection subsets, fractional direction layers,
  alpha to 1.5) and is the **current best artifact**; version_C adds real heretic in the
  training loop (`experiments/version_c_loop.py`) and is a **regression — do not build on it
  without reading handoff section 0**.
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

## Repo layout (reorganised 2026-08-02)
Nothing lives in the repo root but `setup.sh`. Scripts are grouped by purpose under
`scripts/{setup,runs,eval,probes,tools,archive}/` — see `scripts/README.md`. Logs are grouped
under `logs/{training_runs,heretic,eval,probes,panels,drivers,ops}/` — see `logs/README.md`.
`logs/panels/` and `scripts/archive/` hold known-untrustworthy v7/v8-era output; do not reuse.
Day-to-day: `python scripts/tools/tf_status.py` and `python scripts/tools/run_monitor.py -w`.

## Hard conventions (do not violate)
- **Guard on the artifact, not the directory.** `[ -d results/X ]` is true for a directory a
  killed job left empty; the rerun then skips and the judge reads a file that never existed.
  This silently destroyed four eval arms on 2026-08-01/02. Guard on `generations.jsonl`,
  `summary.json`, `results_*.json`, `model.safetensors`.
- **Never `pkill -f` a global pattern** — it killed three concurrent jobs. Scope to your own
  child: `pgrep -P "$pid" -f 'VLLM::EngineCore'`.
- **Never scale a direction layer between architectures.** Qwen peaks at L20/28, Llama at
  L13/16; the proportional guess (L11) is a local MINIMUM. Sweep the base model and measure.
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
  stale `scripts/tools/push_to_hf.py` uploader.
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
