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

## FT-defense attempts (all P4) — NO WIN yet, crux found

`experiments/train_ft_resistant{,_v4,_v4_scaled,_v5}.py`. TAR-style: simulate
attacker FT in an inner loop, shape θ so the post-FT model stays safe.
- v2/v3 (comply-CE-up, mlp): "held" K=1 but broke K≥5 — an artifact of a weak
  beatable inner, not real robustness.
- v4/v5 (generation objective: greedy-gen at θ', pull-refusal + unlikelihood;
  all-scope; kv-cached gen): K=1 0.80/0.70 — regressed/no win.

**CRUX (locked):** the first-order SGD inner sim makes a θ' that REFUSES in
generation (frac_comply=0) while comply_ce is low — but the real 5-epoch AdamW
attack makes a model that COMPLIES in generation. Inner sim breaks teacher-forced
CE, not generation → we defend the WRONG θ'. Objectives never engage.

## Frontier (judge ASR, AdvBench n=200; re-run at 520 for pub)

| K | v5 | v4(all) | v3 | v2 | v7 | base |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.695 | 0.805 | 0.005 | 0.000 | 0.740 | 0.800 |
| 5 | 0.650 | 0.590 | 0.655 | 0.670 | 0.725 | 0.660 |

## Next (fresh box) — see `docs/research_directions_2026_07_02.md`
Chosen program: **capability moat around the safe basin** (pure open-weight, target
abliteration-resist + FT cost ≥ SOTA dozens–hundreds). Build order:
1. **TAR done right (Patcher-style)** — real AdamW all-param inner + generation
   objective [fixes our inner-sim≠attack crux]. Re-sweep K∈{1..100}.
2. **MAD-on-the-gradient** — finite-diff: any comply-FT step must hurt capability.
3. **Rep-rerouting + deepen safety** (circuit-breaker style), compose with v7.
4. **Moonshot:** engineer loss landscape (sharpness asymmetry / mode-connectivity /
   reachability regularizer) so AdamW can't crawl out of safe basin.
Metrics: judge-ASR vs K to 100+, abliteration battery, adaptive-attacker sweep,
MMLU/GSM8K. Then OBLITERATUS (P2), seeds, Qwen/Llama.

## Infra
- Judge locally: `~/miniforge3/envs/env_ml/bin/python experiments/judge_generations.py
  --num-workers 64 --judge-max-tokens 512` (DeepSeek V4 Flash, needs `.env`).
- Box `vast_tamperforge` (`/venv/main` python) — torn down after this session.
  Box git = private HTTPS, needs PAT in remote to push. Never `nohup` box cmds.
- Models on private HF `aaronrockmenezes/tamperforge` (`scripts/push_to_hf.py`).
