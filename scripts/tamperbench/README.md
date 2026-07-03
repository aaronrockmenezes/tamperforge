# tamperbench — consolidated evaluation harness

Two driver scripts over one 3-model registry (`models.sh`): gemma-3-1b, Qwen3-0.6b,
Llama-3.2-1b, each with its ABL-v7 forged ckpt + per-model direction-layer.

Forged ckpts live on private HF — pull first on a fresh box:
```bash
bash scripts/tamperbench/pull_ckpts.sh          # gemma/qwen/llama v7 .pt into outputs/
```
Judge is LOCAL (never on box): after a run, pull `results/<run>/generations.jsonl` and
`~/miniforge3/envs/env_ml/bin/python experiments/judge_generations.py --generations ... --num-workers 64 --judge-max-tokens 512`.

## 1. tamperbench.sh — product credibility (6 clean models)
Does the forged model behave like a normal safe assistant when NOT attacked?
- Per model: BASE-clean + FORGED-clean → AdvBench 520 refusal + ARC/MMLU-12 capability.
- **Win:** forged refuses AdvBench like base (low harmAct) AND cap ~= base = "free product".
- Known limit (gemma): clean prefill hole + off-AdvBench gibberish; GSM8K −35% (findings).
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/tamperbench/tamperbench.sh          # all 3
ONLY=qwen CUDA_VISIBLE_DEVICES=0 bash scripts/tamperbench/tamperbench.sh
```

## 2. robustbench.sh — attack robustness (the MAD result)
Abliterate each model, measure harm + gibberish + capability. Two attacks:
- **A1 rank-1 all-layer** (diffmeans @ DL) — the cheap standard attack.
- **A2 per-layer adaptive** (`--per-layer`, independent dir every layer) — the strong attack.
- **Win:** base_att/base_pl → high coherent harmAct (cap intact); v7_att/v7_pl → ~0 harmAct
  + gibberish + capability crater. Holds under BOTH attacks = robust to adaptive abliteration.
- `OFFDIST=1` adds HarmBench 200 + BeaverTails 1483 on forged rank-1 (gemma/Qwen hold,
  Llama fails — the diffuse-safety boundary, findings §5b).
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/tamperbench/robustbench.sh
ONLY=llama OFFDIST=1 CUDA_VISIBLE_DEVICES=0 bash scripts/tamperbench/robustbench.sh
```

## Other benchmarks worth adding (not yet scripted)
Priority order for paper completeness:

1. **externalbench** — professional attackers as a fair harness across all 3 models.
   OBLITERATUS (7 methods) + Heretic (KL-Pareto) → materialize → our p0+judge+cap.
   Currently gemma-only + OBLITERATUS-on-gemma inconclusive. Wrap `obl_venv`/`heretic_venv`
   calls per model, decline telemetry/upload. Closes "we only tried our own attack".

2. **ftbench** — fine-tune-resistance frontier (the DEAD thread, but needed as the honest
   negative + methodology figure). `ft_attack.py --n-shots {0,25,50,100,200}` on base vs
   forged vs FTR-v6, judged harmAct + ARC/MMLU. Shows forged = base (no FT-resistance) and
   FTR-v6 "0 harm" = lobotomy (cap ~= chance). Anchors the judge-not-ASR contribution.

3. **prefillbench** — non-gradient jailbreak: `prefill_attack.py` (compliant-prefix forcing)
   on all 6. The clean forged model's honest weakness (gemma clean prefill ASR 0.32 > base).
   Distinct axis from abliteration — a jailbreak that doesn't touch weights.

4. **genbench** — generative capability the MC benches miss. GSM8K (already know gemma clean
   −35%, abliterated −98%) + a coding/IFEval slice. MC (ARC/MMLU) hides chain-of-thought
   collapse; this is where the "free product" claim is weakest and most honest to show.

5. **layerbench** — per-layer refusal sweep as a reusable driver (finish gemma 5/26 → 26/26).
   Loop L, rank-1 ablate @L, AdvBench 200, judge → per-model DL profile figure.
   Validates DL selection (the contribution) + is the current open box-task.

6. **scalebench** — model-scale ladder (SmolLM2-1.7B, Ministral-3B, Phi-4-mini, Nemotron-4B
   with `--optim adamw8bit`; MoE OLMoE, hybrid nemotron_h need code work). Does MAD hold as
   params grow / architecture changes? The generality claim beyond 3 sub-2B models.

7. **seedbench** — n≥3 reproducibility: train k seeds/model, report win-rate + variance.
   Names the seed-fragility limitation quantitatively instead of anecdotally (n=2 gemma).
