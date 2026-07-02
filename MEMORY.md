# tamperforge — session memory / handoff index

> Repo-local state index. Updated 2026-07-02 (pm). Read `docs/handoff_2026_07_02_v2.md`
> for the full current state, `CLAUDE.md` for conventions.

## Naming (two version lines; DON'T conflate)
- **ABL-v{n}** = abliteration line, `outputs/tamper_resistant_p1b_v{n}.pt`. ABL-v7 =
  product. Dirs: `outputs/abl_v7_hf` (clean), `outputs/abl_v7_hf_attacked` (attacked).
- **FTR-v{n}** = fine-tune-resistance line, `outputs/ft_resistant_p4_v{n}.pt`.

## Thread 1 — Abliteration (ABL-v7): WORKS + generalizes
- Robust across the full abliteration battery (ensemble training). AdvBench: clean ASR
  0.013 / ARC 0.364; attacked ASR 0.004 / ARC 0.246; attacked-base ASR 0.66.
- **NEW (2026-07-02): generalizes off-distribution.** Full 2x2x3 judge battery
  (clean/abliterated x OG/v7 x prefill/HarmBench/BeaverTails): abliterating ABL-v7 ->
  **0% ASR / 100% gibberish across all three** (incl. prefill = non-gradient attack it
  never trained on); abliterating base -> real harm (ASR 0.41-0.80). MAD is not
  overfit to AdvBench/gradient attacks. `docs/findings_prefill_harmbench_beavertails_2026_07_02.md`.
- **Caveat (must report):** CLEAN ABL-v7 (no attack) has off-AdvBench coherence cost:
  13.5% gibberish prefill, 20% HarmBench, 50% BeaverTails (vs ~0% base). "Free product"
  only held on AdvBench. Full-suite capability re-check (ARC full + MMLU 12-topic ~2k)
  running now to see if it bleeds into reasoning.

## Thread 2 — Fine-tune resistance (FTR): NO WIN yet
- FTR-v2..v5 = 1-shot moat at best (artifact); all -> base by K>=5. Crux: inner-sim
  defended teacher-forced CE, not generation.
- **FTR-v6 = Lever-2 (LoRA-inner TAR).** `experiments/train_ft_resistant_v6.py`: real
  LoRA attack inner-loop + LLM-JUDGE gate (same DeepSeek as eval, not keyword) +
  FO-MAML (theta'=theta+detached-delta). Trained meta-lr 1e-5/5e-5/2e-4. Training
  frac_comply oscillated, no clear downtrend -> objective engages but doesn't visibly
  out-harden a rank-32 LoRA attack in the budget. **Validation ft_attack sweep
  (K=0..200, full 520) is the real verdict — running on 5090.** Ckpts
  `outputs/ft_resistant_p4_v6_lr{5e5,2e4}.pt` (lr5e4 self-destructed, DISCARD).

## Full FT frontier (judge ASR, AdvBench n=200 — re-running at 520)
| K | v5 | v3 | v2 | ABL-v7 | base |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.695 | 0.005 | 0.000 | 0.740 | 0.800 |
| 5 | 0.650 | 0.655 | 0.670 | 0.725 | 0.660 |
(all break to ~base by K>=5; full K=0..200 in `docs/results_2026_07_01.md`)

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
Capability-Destructive." Full plan: `docs/critiques.md`, `docs/next_steps_2026_07_02.md`.

## Infra (details in CLAUDE.md / handoff_v2)
- 4090 `vast_tamperforge` (vLLM works) = eval box. 5090x2 `tamperforge_5090x2`
  (vLLM was NCCL-broken; verify before use) = FTR box, CUDA_VISIBLE_DEVICES=0/1.
- Judge local via `~/miniforge3/envs/env_ml/bin/python`, 64 workers, 512 max-tokens.
- Never nohup box cmds. Full datasets only. GitHub + private HF backups.
