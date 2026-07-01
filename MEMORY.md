# tamperforge — session memory / handoff

> Repo-local state index. Last updated 2026-07-02. Full detail: `docs/`, `ROADMAP.md`.

## Where the project stands

- **P0/P1/P1b-A: done.** v7 (`outputs/tamper_resistant_p1b_v7.pt`) = robust vs the
  full abliteration attack battery (ensemble training closed all leaks). Clean
  product ~free. See `docs/results_2026_07_01.md`.
- **P4 attack (`experiments/ft_attack.py`): v7 breaks at K=1 fine-tuning.**
  judge ASR 0.013→0.740 ≈ base. Abliteration-robust ≠ FT-robust.
- **P4 defense (`experiments/train_ft_resistant.py`, TAR-style, warm-start v7):**
  v2 & v3 both = **1-shot moat** — resist K=1 (ASR ~0.00, coherent refusals),
  break K≥5 (~0.66 = base). Ckpts `outputs/ft_resistant_p4_v{2,3}.pt`.
  - **Root cause (locked):** objective defends `comply_after` = teacher-forced
    comply-CE, which does NOT stop post-FT greedy GENERATION. proxy≠generation, 4th
    time. Not a hyperparam fix.

## Active work: v4 — generation-level FT-defense

`experiments/train_ft_resistant_v4.py` (WIP). Meta-objective on POST-FT GENERATION:
after simulated inner FT (θ'), greedy harmful gen must stay refusal/gibberish
(argmax-divergence at θ', not comply-CE). Plan: `docs/ft_resistance_plan.md`.

## Frontier (judge ASR, AdvBench n=200; re-run at 520 for pub)

| K | v3 | v2 | v7 | base |
|--:|--:|--:|--:|--:|
| 1 | 0.005 | 0.000 | 0.740 | 0.800 |
| 5 | 0.655 | 0.670 | 0.725 | 0.660 |
| 10 | 0.625 | 0.660 | 0.590 | 0.600 |
| 25 | 0.710 | 0.660 | 0.725 | 0.680 |

## Next
v4 (generation-level) → re-sweep at 520 → OBLITERATUS (P2, AGPL harness on v7) →
seeds + Qwen/Llama + MMLU-full/GSM8K → RepNoise complement.

## Infra
- Judge locally: `~/miniforge3/envs/env_ml/bin/python experiments/judge_generations.py
  --num-workers 64 --judge-max-tokens 512` (DeepSeek V4 Flash, needs `.env`).
- Box `vast_tamperforge` (`/venv/main` python) — torn down after this session.
  Box git = private HTTPS, needs PAT in remote to push. Never `nohup` box cmds.
- Models on private HF `aaronrockmenezes/tamperforge` (`scripts/push_to_hf.py`).
