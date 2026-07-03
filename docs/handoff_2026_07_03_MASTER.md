# MASTER HANDOFF — 2026-07-03 (multi-model + attack-robustness campaign)

Read order for the next agent: THIS → `docs/findings_multimodel_adaptive_2026_07_02.md` →
`docs/findings_prefill_harmbench_beavertails_2026_07_02.md` → `docs/devlog_2026_07_02.md`
(phases 1–3) → `CLAUDE.md` → `MEMORY.md`. This supersedes older handoffs for the
abliteration thread's current state.

## TL;DR of the whole project
tamperforge entangles safety with capability so cheap uncensoring self-defeats
("smart-and-safe XOR dumb-and-dangerous"). Two threads:
- **ABL (abliteration-resistance) = the WINNING thread.** ABL-v7 recipe. Generalizes across
  3 architectures + survives adaptive + professional attacks (with one honest boundary).
- **FTR (fine-tune-resistance) = DEAD.** v2–v6 + FTR-TAR all failed (lobotomy or can't
  out-harden). Do NOT reopen without a fundamentally new lever. Paper anchors on ABL.

## THE HEADLINE RESULTS (all LLM-judged, DeepSeek V4 Flash; harmAct = coherent harm)
### 1. ABL-v7 works on gemma-3-1b (product), reproduced across 2 seeds
Ablate ABL-v7 → ~0 ASR / ~100% gibberish AND capability crater (ARC 0.35→0.27, MMLU
0.40→0.27, GSM8K 0.26→0.006). Ablate base → coherent harm (0.66–0.82) capability intact.
### 2. Generalizes to 3 architectures (per-model refusal-layer selection)
| model | ablate-v7 harmAct (AdvBench) | v7_att gib | capability | verdict |
|---|--:|--:|---|---|
| gemma-3-1b (seed 42, 2) | ~0.00 | ~1.00 | crater | FULL WIN |
| Qwen3-0.6b (DL 14) | 0.00 | 0.99 | no crater | CLEAN gibberish-wall |
| Llama-3.2-1b (DL 13) | 0.13 | 0.74 | no crater | PARTIAL (leaky) |
### 3. Survives a PER-LAYER ADAPTIVE attack (our `--per-layer`, 26–28 indep dirs) — 3/3
| model | base per-layer harmAct | v7 per-layer harmAct |
|---|--:|--:|
| gemma | 0.712 | 0.000 |
| Qwen | 0.565 | 0.002 |
| Llama-L13 | 0.588 | 0.058 |
### 4. Off-distribution (HarmBench/BeaverTails): gemma+Qwen HOLD, Llama FAILS
Qwen v7-ablate → 0.000/0.001 harm (gibberish). Llama v7-ablate → **0.42/0.44 harm** (≈base;
worse than base on BT). Boundary: diffuse-safety models (Llama) → AdvBench-overfit; the
method generalizes off-dist only for concentrated (gemma) or broad-wall (Qwen) safety.
### 5. Professional external attackers on gemma-v7 (running / partially done)
Heretic (KL-minimizing Optuna auto-abliterator) Pareto front on gemma-v7: **0 refusals needs
KL 0.70 (wrecked model)**; KL-lossless (0.004) leaves 79/100 refusals. The attacker's OWN
optimizer can't decensor without destroying capability = MAD proven externally. OBLITERATUS
7 methods + Heretic 3 Pareto points: our-eval verdict PENDING (that's the current running work).

## KEY THAT VARIES BY MODEL (the DL-selection contribution)
Codex picked direction-layer by 50%-depth GUESS → WRONG for Llama (base uncensors 0.60@L13
vs 0.47@L8; retrain@L13 cut harmAct 0.65→0.13). Qwen guess (14) was also suboptimal (peak
L20 0.38 vs 14=0.22) but Qwen defended anyway (gibberish-wall is layer-robust). **Full-layer
judged sweeps:** Qwen done (peak L20), Llama running, gemma running. Pick DL by the actual
per-layer refusal profile (judged harmAct, NOT keyword).

## EXACT RECIPES / COMMANDS FOR KEY RUNS
### ABL-v7 training (canonical, from `results/p1b_a_ensemble_v7/manifest.json`)
```
python experiments/train_tamper_resistant.py --model-id <MID> --out <ckpt> \
  --train-scope all --abliterate-layers all --attack-ensemble --direction-layer <DL> \
  --recompute-direction-every 25 --gib-mode argmax --gib-gen-tokens 32 --gib-gen-prompts 2 \
  --lambda-gib 4 --lambda-uncensor 4 --lambda-safe 1 --lambda-reg 0.1 \
  --steps 500 --eval-every 25 --lr 1e-5 --seed <S>
```
Per-model DL: gemma 13, Llama 13, Qwen 14 (Qwen sweep-peak 20 but 14 worked). gemma-1b/llama-1b/
qwen-0.6b fit full-prec 24GB. >1.3B needs `--optim adamw8bit`. gemma needs eager attn (auto).
Scripts: `scripts/multimodel_abl_v7/run_{qwen3_1p7b,llama32_1b,gemma3_1b_v7_seed}.sh`.
### Battery (per model): `scripts/multimodel_abl_v7/run_battery.sh`
`MID=.. CKPT=.. DL=.. TAG=.. CUDA_VISIBLE_DEVICES=.. bash scripts/multimodel_abl_v7/run_battery.sh`
Materializes base-att/v7-clean/v7-att, gens AdvBench 520@512 + lm_eval ARC/MMLU. Judge locally.
### Per-layer adaptive attack (OUR reimplementation of adaptive abliteration)
`save_p1b_checkpoint.py --model-id <MID> [--checkpoint <v7.pt>] --attack all --per-layer --out <dir>`
### Refusal-layer sweep (find true DL): loop L, `save_p1b_checkpoint --attack all --direction-layer $L`,
p0 gen advbench 200, JUDGE (keyword lies — see gotchas).
### External attackers (AGPL, separate venvs, NEVER vendor, no telemetry)
OBLITERATUS: `obliteratus obliterate <model> --method {basic,advanced,aggressive,surgical,optimized,inverted,nuclear} --dtype bfloat16 --gpus <G> --output-dir <dir>` (NO --contribute).
Heretic: `heretic --model <model> --winsorization-quantile 0.95 --export-strategy merge` (decline upload; pick Pareto trial). Eval BOTH outputs with our p0+judge+lm_eval.
### Judge (LOCAL): `~/miniforge3/envs/env_ml/bin/python experiments/judge_generations.py
  --generations <dir>/generations.jsonl --run-id X_judged --num-workers 64 --judge-max-tokens 512`

## PROBLEMS FACED + SOLUTIONS (condensed; full detail in devlog phases 1–3)
1. Leaked-GPU vast box → destroy+re-rent; verify `nvidia-smi` ~0 used first.
2. Qwen-1.7B OOM (fp32 AdamW >1.3B on 24GB) → swapped to Qwen3-0.6B; added `--optim adamw8bit` (default off).
3. **gemma NaN under sdpa** → `load_model` forces `attn_implementation=eager` for gemma (env `TF_ATTN_IMPL`).
4. **Seed fragility** (~half gemma seeds fail): argmax gib-CE spiked to inf in bf16. Fixed CE
   in **fp32 + clamp(30)** (kills NaN crashes) BUT some seeds still don't form the basin →
   genuine limitation. n=2 (seed42, seed2). grad-clip default 1e9 (OFF — tight clip throttled gib).
5. **keyword ASR unreliable BOTH ways** (Qwen v7 kw0.99/judge0.01; Llama base kw0.89/judge0.60)
   → ALWAYS LLM-judge; keyword only coarse-ranks base uncensoring.
6. vLLM startup OOM in tight loops → `--vllm-gpu-memory-utilization 0.85` + `--max-length 4096`.
7. SSL `UNEXPECTED_EOF` on HF dataset HEAD = HARMLESS (cached fallback).
8. Background `( )&` inside a tool call gets killed on return → re-judge as proper bg task.
9. FailSpy/abliterator = not on PyPI + TransformerLens/no-HF-export → dropped (method = our rank-1/per-layer, already covered).

## ABL-v8 (2026-07-03) — CONDITIONAL WALL, fixes v7's clean tax (NEW WINNING RECIPE)
`experiments/train_tamper_resistant_v8.py`. v7 problem: clean model degraded (Qwen v7-clean =
97% gibberish on harmful, IFEval −30%; gemma −62%, llama −89%). Root cause: v7's clean anchor
(`L_task`, teacher-forced CE) is loglikelihood-style → misses free-gen drift, so `λ_gib` bled
into clean generation. **v8 adds a GENERATIVE clean anchor** `L_clean_gen = λ_clean·KL(base_gen‖
clean_gen)` on benign + eval-in-loop 24-prompt IFEval probe. Single-stage λ traces a clean↔wall
Pareto (no both-high point). **Fix = TWO-STAGE curriculum** (`--clean-start-step/--clean-ramp-
steps/--stage2-lambda-gib`): form wall first (λ_clean 0), then ramp clean anchor to repair.
Qwen result (AdvBench-judged, IFEval): clean gib **97%→1.5%**, clean IFEval **0.218→0.305 (=base
0.312)**, attack wall **held 96% gib**, attack IFEval craters 0.122. **v8 strictly dominates v7
on the clean side, no wall loss.** Recipe (Qwen): DL 20, λ_gib 8, λ_clean 3, clean-start-step 250,
ramp 100, stage2-λ_gib 4, 500 steps. TODO: replicate on gemma/llama (harder); n=1 seed.
Full matrix: `scripts/eval_matrix_qwen.sh` (base/v7/v8 × clean/att × AdvBench/HarmBench/SR/JBB/
SORRY + ARC/MMLU/IFEval/GSM8K), running. **v8_att is a DEEPER MAD than v7_att:** GSM8K crater
v8 0.401→0.020 (−95%) vs v7 0.376→0.335 (−11%) — v7's wall is harm-narrow (stays capable),
v8 destroys general capability under attack.
**NEW-BENCH TODO (HIGHEST-PRI) — `docs/todo_new_benchmarks_HIGHPRI.md`:**
add XSTest + OR-Bench-Hard-1K (over-refusal, the axis we have ZERO coverage on) + MT-Bench +
MBPP + **SimpleQA (1k subset)** now; **MultiBreak DEFERRED** (run later). SimpleQA-1k + MultiBreak
are the two explicitly-flagged high-pri items.

## v8 TRAINER IMPROVEMENT IDEA (TODO — build after llama run)
Currently v8 saves only the FINAL step (step-500), which is a NOISY single-batch snapshot —
Qwen's step-500 looked weak (gib_ce 1.24) but the materialized ckpt was 96% wall (saved by luck).
**A. Save BEST ckpt (not final):** each eval step, score ablated-harm (want low) on ~50 held-out
harmful + clean-IFEval-probe (want high); keep the best-on-both ckpt. Fixes the snapshot lottery
universally. **B. Stage-1 reset-to-best-wall:** at `clean_start_step`, reset weights to the
strongest-wall stage-1 ckpt before repair. CAVEAT: strongest wall may be HARDEST to repair
(deeper gibberish) — the knee wall may beat the extreme; A-style best-FINAL selection sidesteps
this. Build A (+ optional B) into `train_tamper_resistant_v8.py` (~30 lines: held-out ablated-harm
eval each step + track best). Use for gemma v8 + any re-runs.

## EXISTING ISSUES / OPEN
- **GEMMA FULL-LAYER SWEEP DONE (2026-07-03) — all 26/26 judged.** Base-ablation harmAct per
  layer: peak **L14=0.890**, trained **DL13=0.845** (both in the L13–15 peak band); refusal
  concentrated mid-late (L12–17), early layers inert (≤0.05), L6 breaks model (100% gib), small
  L25 echo (0.51). **Validates DL=13** = in the peak band. Qwen (peak L20=0.38) + Llama
  (peak L13=0.565) already complete. All 3 per-model refusal profiles now judged →
  DL-selection figure ready. `results/sweep_gm_L{0..25}_adv200(_judged)`.
- **Heretic base-ref DONE (2026-07-03) — confound CLOSED.** Heretic on base gemma → 93% coherent
  harm @ KL 0.098, cap intact (vs v7: can't uncensor w/o wreck). See
  `docs/findings_external_benches_ifeval_2026_07_03.md` §1. Supersedes OBLITERATUS-inconclusive.
- **External prompt benches DONE (2026-07-03, Tier-1)** — StrongREJECT/JBB/SORRY-Bench, clean v7
  ≥ base safety + clean-gibberish tax 14/18/41%. TODO Tier-2 = official judges on Blackwell +
  extend to Qwen/Llama v7. `scripts/external_benches/`, findings doc §2.
- **IFEval DONE (2026-07-03)** — clean v7 generative tax −62% (prompt_strict 0.542→0.205); MAD
  split holds (base_att 0.530 ≈ base; v7_att 0.120). NEW honest limitation to foreground. §3.
- **TamperBench refusal_ablation — TODO Blackwell** (OOM on 24GB fp64 alloc). Turnkey:
  `docs/todo_tamperbench_blackwell.md`. Third-party standardized version of the abliteration result.
- **Llama off-dist FAILURE** (0.42 harm) — honest limitation, documented.
- **Seed fragility** — n=2 gemma; fp32 fix didn't fully de-lottery. Candidate: gib warmup / higher early λ_gib.
- **Clean generative tax (λ_gib knob)** — IFEval/GSM8K show clean v7 degraded on free-form gen;
  a gentler λ_gib / gib-warmup could lift it at a robustness cost. Optional sweep.
- **External-attacker eval on Qwen/Llama PENDING** — OBLITERATUS/Heretic on the other archs (gemma done).
- **MoE + hybrid untested** — `--per-layer`/attack code is `self_attn`+`mlp`-specific; MoE routes through experts, hybrid (nemotron_h) has mamba layers w/o self_attn → needs code changes. Phase-2 mini-project.
- Rigor debt: single seed for Qwen/Llama; off-dist only AdvBench-judged for some.

## NEXT STEPS (priority)
1. **Judge the external-attacker matrix** (OBLITERATUS 7 + Heretic 3 + base refs) → the flagship robustness table. If all hold → paper-ready.
2. Full-layer sweeps finish (gemma/llama) → per-model DL figure.
3. External attackers on Qwen + Llama-L13 (esp. Llama, the weak link).
4. **Scale the model ladder** (blog range) — immediate 4 with `--optim adamw8bit`:
   `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `mistralai/Ministral-3-3B-Instruct-2512`,
   `microsoft/Phi-4-mini-instruct`, `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (hybrid=experimental).
   Larger ladder (250M–9B, MoE, hybrid): see `MEMORY.md` / this session. 7–9B needs A100.
5. Seed-robustness fix (optional, for clean n≥3).
6. Draft paper: anchor = generalizes across archs + survives adaptive + professional attackers;
   contributions = per-model DL selection, judge-not-keyword, MAD-via-attacker's-own-KL-optimizer;
   honest limits = Llama off-dist, seed fragility, diffuse-safety boundary.

## INFRA
- Box: `tamperforge_3090x4` (ssh5.vast.ai:30637 / direct 120.238.149.205:33175), 4x RTX 3090 24GB,
  `/venv/main` (torch2.11+cu130 vLLM0.24). External-attacker venvs: `/workspace/obl_venv`,
  `/workspace/heretic_venv` (AGPL, separate). Logs → `/workspace/logs` (tee).
- Judge LOCAL via `~/miniforge3/envs/env_ml/bin/python`, 64 workers, OPENROUTER_API_KEY in .env.
- GitHub `aaronrockmenezes/tamperforge` (code/docs/results). Private HF same name (.pt + dirs,
  `scripts/push_to_hf.py` — extended for multimodel ckpts). OG gemma v7: `outputs/hf_og/adapters/tamper_resistant_p1b_v7.pt`.
- CONVENTIONS: never nohup box cmds (user watches, tmux); full datasets no subsets; judge-not-keyword;
  HF private; never vendor AGPL attacker code (separate harness + cite).

## KEY CKPTS
`outputs/tamper_resistant_p1b_v7.pt` (gemma OG ABL-v7, HF-backed), `..._p1b_v7_seed2.pt`,
`..._qwen3_0p6b_v7.pt`, `..._llama32_1b_v7_L13.pt` (the good Llama). FTR (dead): `ft_resistant_p4_v6_*`,
`ft_resistant_p4_tar_*` — DISCARD.
