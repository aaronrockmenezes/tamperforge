# Compaction brief — tamperforge version_A/version_B, 2026-08-01

Paste this into a fresh context to resume. Durable findings live in
`docs/handoff_2026_08_01_version_a_b.md`; this file is IN-FLIGHT state only.

## Where things run
- Box: `ssh vast-versiona-3090` (ssh2.vast.ai:13645), repo `/workspace/tamperforge`
- Always: `source /venv/main/bin/activate && set -a && . ./.env && set +a`
- Local repo `~/Desktop/Projects/Mech Interp/tamperforge`; judge locally via
  `~/miniforge3/envs/env_ml/bin/python`
- `/workspace` is NOT a volume — pull results off as you go, the box dropped twice
- Budget ~$4.50 of ~$5 spent at $0.232/hr

## Running now
tmux `fin3` -> `chain2.sh`, log `logs/training_runs/final_chain2.log`.
Stage 1 (IFEval x 10 arms) DONE. Stage 2: heretic-vs-version_B trials t17/t99/t65,
each getting p0_baseline_eval (520 walledai, thinking-off) + judge + lm_eval
ARC / MMLU-12 / GSM8K-5shot / IFEval.

Helper scripts already on the box: `vb_report.py`, `ifev_report.py`, `dircmp.py`,
`final_report.py` (adapt paths to `results/hvb_t*_judged`, `results/hvbcap_t*_*`).

## What stage 2 answers
version_A conceded **0.2615 harmful with GSM8K 0.4064 intact** to heretic t175.
- **t65 is the one to watch**: matched KL (0.0138 vs 0.0124) and reads from
  direction layer 14.1 — the low band version_B was supposed to close. Harmful +
  intact GSM8K there means the escape route is still open.
- t99 = matched refusals (3/100). t17 = max uncensoring at KL 0.1397; its chat
  output was `\boxed{}` degeneration loops (n=1, anecdote).

## Reference table (export path, 520 prompts, GSM8K = STRICT)
| model / attack | harmful | GSM8K |
|---|---|---|
| v8 clean | 0.0115 | 0.4049 |
| v8 + surgical k16 | 0.4365 | 0.2851 |
| version_A clean | 0.0058 | 0.3980 |
| version_A + rank-1 | 0.0000 | 0.0447 |
| version_A + surgical | 0.0019 | 0.3844 |
| version_A + heretic t175 | 0.2615 | 0.4064 |
| version_B clean | 0.0019 | 0.3889 |
| version_B + rank-1 | 0.0000 | 0.0091 |
| version_B + surgical | 0.0000 | 0.1054 |

## Non-negotiable gotchas
1. GSM8K: use **strict-match**. flexible-extract takes the LAST number and these
   models ramble past the answer.
2. Use the **export path**, not `version_a_eval_panel` — it reported v8 surgical at
   0.078 against a known 0.448 on bit-identical weights. Unexplained.
3. **Run a known-answer control before trusting any harness.** Three bugs were
   caught only this way, all producing plausible numbers.
4. vLLM hangs AFTER lm_eval writes results. Use the background+poll+kill-EngineCore
   pattern from `scripts/run_mad_v10_s175_vllm_caps.sh` (`1a4e603`); `chain2.sh`
   copies it. Never `pkill -9` — orphans an EngineCore in the HOST pid namespace and
   leaks VRAM irrecoverably.
5. `pkill -f <pat>` matches your own command line.
6. **IFEval is not a capability measure here** — it rises under attack (version_B
   rank-1 0.3573 vs clean 0.3213) while GSM8K craters. It scores format compliance,
   which survives semantic collapse.

## Already backed up (do not redo)
GitHub through `f55cfe7`. Private HF `aaronrockmenezes/tamperforge`:
`version_a_2026_07_31/` and `version_b_2026_08_01/`, s400+s450+s500 each.

## Next after stage 2
1. Report version_B-vs-heretic; commit + push.
2. **version_C = attack-in-the-loop.** version_B is still random sampling against an
   optimiser; 500 draws cannot cover a 9-param TPE search. Short heretic pass (20-30
   trials) against current weights every ~100 steps, cache winners, sample alongside
   random attacks. ~10 min added per run.
3. Do NOT claim "tamper-resistant". Honest claim: MAD raises attacker cost from one
   line of code to a 200-trial optimisation forced to accept 5x the perturbation.
