# TamperForge — Current TODO

> **Decision order:** break ABL-v8 with stronger attacks before spending on model scale.
> The published claim remains limited to attack-cost shaping against abliteration;
> fine-tune resistance is closed negative work.
>
> **Updated 2026-07-18.** The "break v8 with stronger attacks" line item above just
> happened for real: Heretic (adaptive KL-optimizing abliteration) breaks ABL-v8 on
> Llama-3.2-1B — 40% coherent harm at zero capability cost mid-strength, 88% at its
> strongest trial. See `docs/heretic_v8_llama_2026_07_18.md` + `docs/devlog_2026_07_17.md`.
> This reprioritizes everything below: finding out whether this is Llama-specific or hits
> gemma/Qwen v8 too is now the single highest-priority open question, above rank-k SVD or
> surgical ablation (which haven't cracked anything yet — Heretic just did).

## Top priority right now

- [ ] **Run Heretic against gemma v8 and Qwen v8** (same recipe as Llama: Pareto-trial pick
  spanning weak/mid/strong KL budget, full AdvBench+ARC+MMLU+IFEval+GSM8K matrix). This
  answers architecture-specific vs product-specific before anything else matters.
- [ ] If gemma/Qwen v8 also crack: this becomes the paper's central adaptive-attack finding,
  not a footnote — may require revisiting the training-time attack simulation (Heretic's
  per-projection parameterization isn't covered by the current rank-1 `_sample_attack`).
- [ ] If gemma/Qwen v8 hold: Llama's crack is explained by its already-known-shallow crater
  (this TODO's old Llama row, below) — still needs stating honestly in the paper, but doesn't
  sink the whole claim.

## Models to test now

| Priority | Model | What we run now | Gate / reason |
|---|---|---|---|
| P0 | `google/gemma-3-1b-it` | Heretic adaptive attack (weak/mid/strong KL trials) | Unknown whether Llama's crack generalizes — highest-value open question. |
| P0 | `Qwen/Qwen3-0.6B` | Heretic adaptive attack; five prospective ABL-v8 seeds; rank-k SVD and surgical-ablation attack Pareto | Strongest current v8 result on the naive attack; still untested against Heretic. |
| P0 | `meta-llama/Llama-3.2-1B-Instruct` | Rank-k SVD, surgical, and per-layer adaptive attacks; complete extended suite | **Confirmed cracked by Heretic** (2026-07-18) — the hardest current family, now confirmed hard for real. |
| P1 | `google/gemma-3-1b-it` | `S2GIB=8` ABL-v8 rerun (separate from the Heretic item above) | Close its remaining −15% clean IFEval residual before treating 3/3 as equally strong on the clean side. |
| P1 | `microsoft/Phi-4-mini-instruct` (3.8B) | Direction-layer sweep, then ABL-v8 only if P0 attack gate holds | First meaningful scale test; requires the 96GB box. Do not start until the Heretic question above is answered — no point scaling a defense that might not survive adaptive attacks. |
| P2 | `mistralai/Ministral-3-3B-Instruct-2512` | Same sweep/train/eval protocol after Phi | Second architecture at useful scale; do not start until Phi and P0 are clean. |
| P2 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | Optional low-cost ladder point / parallel seed work | Fits 24–32GB with `adamw8bit`; not a substitute for the 3–4B scale test. |

## Must run before scaling claims

- [ ] Heretic on gemma v8 + Qwen v8 (see Top priority above — supersedes the ordering below).
- [ ] Freeze a prospective protocol: selection split, snapshot gates, attack budgets, and full held-out test suite.
- [ ] Build rank-k SVD/subspace ablation and report the best attacker harm-versus-capability frontier.
- [ ] Build surgical refusal ablation and run it on Qwen-v8 and Llama-v8.
- [ ] Add the all-architecture per-layer adaptive matrix for v8, not just the older v7 result.
- [ ] Run Qwen-v8 for five seeds; report every seed and no-survivor outcome.
- [ ] Run the ART baseline on its native protocol before broad comparative claims.

## Parallel closure work

- [ ] Gemma: run `S2GIB=8`, pick by the locked automatic rule, then rerun the full matrix.
- [ ] Llama: run `scripts/eval_matrix_new.sh` for XSTest, OR-Bench, SimpleQA, and MBPP.
- [ ] Finish the TamperBench third-party validation sweep (`docs/devlog_2026_07_17.md` Thread
  2) — only `gemma_base` confirmed clean so far, 5 of 6 runs remain.
- [x] ~~Update the root README, ROADMAP, and AGENTS status blocks~~ — done 2026-07-18.

## Parked, not closed

- [ ] Qwen3-8B thinking-mode scale attempt. Trained successfully (3 OOM iterations, see
  devlog); pick-job's raw generations were destroyed by an infra bug (fixed) and not
  re-run. Resuming needs: fresh pick job against the preserved trace, then the four-cell
  eval. See `docs/qwen3_8b_showcase_walkthrough.md`.

## Explicitly not now

- [ ] Do not reopen FTR/TAR without a fundamentally new mechanism.
- [ ] Do not claim fine-tune resistance, tamper-proofing, or broad model safety.
- [ ] Do not claim "survives Heretic" or "survives adaptive attacks" as a blanket statement —
  true for gemma v7, false for Llama v8, gemma/Qwen v8 unknown. See `docs/related_work.md`
  correction.
- [ ] Do not move to 7B–12B before the Heretic question above + the P0 attack gate pass.
