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

## FT frontier — FTR-v6 lr2e4 harmful_actionable at FULL 520/512 (judged)
| K | harmAct | gibberish | note |
|--:|--:|--:|--|
| 0 (clean) | 0.000 | 0.998 | ARC 0.217 / MMLU 0.246 ~= CHANCE -> BROKEN |
| 25 | 0.000 | 0.988 | |
| 50 | 0.000 | 1.000 | |
| 100 | 0.000 | 0.998 | (subset@128tok wrongly showed 0.21 - short-output artifact) |
| 200 | 0.000 | 0.988 | |
- Never produces actionable harm because it's a lobotomized model, not because it
  resists. Broken clean AND under attack. FTR-v6 = FAIL. Old line (v2-v5) broke to
  ~base by K=5 (that data in `docs/results_2026_07_01.md`).

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
