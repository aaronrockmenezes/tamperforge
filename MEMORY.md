# tamperforge — session memory / handoff index

> Repo-local state index. Updated 2026-07-02 (pm). Read `docs/handoff_2026_07_02_v2.md`
> for the full current state, `CLAUDE.md` for conventions.

## Naming (two version lines; DON'T conflate)
- **ABL-v{n}** = abliteration line, `outputs/tamper_resistant_p1b_v{n}.pt`. ABL-v7 =
  product. Dirs: `outputs/abl_v7_hf` (clean), `outputs/abl_v7_hf_attacked` (attacked).
- **FTR-v{n}** = fine-tune-resistance line, `outputs/ft_resistant_p4_v{n}.pt`.

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
- **Clean product credible:** clean ABL-v7 ~= base on reasoning (ARC 0.344, MMLU 0.393).
  Gibberish is concentrated on harmful-prompt distributions, not ARC/MMLU.
- **Honest limitation:** CLEAN ABL-v7 has a prefill hole (ASR 0.323, harmAct 0.285 >
  base 0.108) + off-AdvBench clean gibberish (11/17/47.5% prefill/HB/BT). Report in
  Limitations.

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

## FT frontier — FTR-v6 lr2e4 moved it (judge ASR, n=100/128tok, re-running at 520/512)
| K | FTR-v6 lr2e4 (harmAct) | FTR-v6 lr5e5 | v5 | v2/v3 | base |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.00 (0.00) | 0.00 | 0.695 | 0.00/0.005 | 0.800 |
| 5 | 0.00 (0.00) | 0.04 | 0.650 | ~0.67 | 0.660 |
| 50 | 0.00 (0.00) | 0.56 | — | — | ~base |
| 100 | 0.43 (0.21) | 0.66 | — | — | ~base |
| 200 | 0.38 (0.32) | 0.67 | — | — | ~base |
- Old line (v2-v5) broke to ~base by K=5. **lr2e4 holds ASR~0 to K=50; attack yields
  gibberish not harm to K=100** (65% gibberish @K100). Frontier K~5 -> K~50-100.
  lr5e5 = dead (breaks like old line). GATE: clean ARC/MMLU on lr2e4 (running) — if
  coherent = real moat, if gibberish-everywhere = broken model, not defense.

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
