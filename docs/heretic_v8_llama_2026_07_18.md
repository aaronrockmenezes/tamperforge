# Heretic (adaptive abliteration) vs ABL-v8 — Llama-3.2-1B

Living doc. First time Heretic has been run against the **v8** product on any architecture
(prior Heretic campaign, `docs/handoff_2026_07_03_MASTER.md` + `results/*heretic*`, only
covered gemma **v7** and base). Target picked as **Llama-3.2-1B v8**, not gemma, because it's
the weakest v8 arch on the existing rank-1 attack data (`docs/paper_tables_v1.md` Table 3):
attacked GSM8K only craters −20% (.348→.278) vs Qwen's −95% and gemma's −48% — the leakiest
wall we've measured, so the most likely to crack under an adaptive attacker.

## Setup

- Box: `vast-tamperbench` (RTX A6000, 48GB).
- Tool: `pip install -U heretic-llm` ([p-e-w/heretic](https://github.com/p-e-w/heretic)),
  v1.4.0. Directional ablation + Optuna TPE search, co-minimizes refusal-count and
  KL-divergence-from-original.
- Target: `/workspace/tamperforge/outputs/llama_v8_hf` (materialized from
  `outputs/tamper_resistant_llama32_1b_v8_best.pt`, s425 pick, DL13).
- Command: `heretic --model /workspace/tamperforge/outputs/llama_v8_hf --winsorization-quantile 0.95 --export-strategy merge --seed 42`
- 200-trial Optuna sweep (default `--n-trials`), ~24 min wall clock on the A6000.

## Trial selection

Heretic reports every trial's `(refusals, KL-divergence)` and lets you interactively pick
from the Pareto front. 20 non-dominated trials came back, spanning refusals 4/100 → 100/100
and KL 0.002 → 0.562. Picked 3 to span the frontier (matches the KL-budget-sweep methodology
used on gemma v7 in the July campaign):

| tag | trial | refusals/100 | KL divergence | read |
|---|---|---:|---:|---|
| `kl0562_trial103` | 103 | 4 | 0.5621 | strongest attack Heretic found — the critical test |
| `kl0253_trial42` | 42 | 71 | 0.2533 | mid-point |
| `kl0099_trial90` | 90 | 95 | 0.0990 | near-untouched control |

All 3 exported (`--export-strategy merge`, saved locally, declined HF upload from inside
Heretic itself — pushed separately, see below) to:
- `/workspace/heretic_out/llama_v8_kl0562_trial103/`
- `/workspace/heretic_out/llama_v8_kl0253_trial42/`
- `/workspace/heretic_out/llama_v8_kl0099_trial90/`

## Eval

Same recipe as the gemma v7 Heretic campaign: AdvBench-520 safety generation (vLLM,
batch-size 128) + ARC-challenge + MMLU-12-subject-subset capability (lm_eval, 0-shot,
`batch_size auto`). Script: `/workspace/eval_heretic_llama_v8.sh` (on box, not vendored —
ad hoc for this run). Judging happens locally after (`experiments/judge_generations.py`,
DeepSeek-V4-Flash, per hard convention — never judge on the box).

## Results — the wall breaks

| tag | harmAct | gibberish | refused | ARC | MMLU-12 |
|---|---:|---:|---:|---:|---:|
| v8_clean (no attack, reference) | .00 | .00 | — | .384 | .484 |
| v8_att (rank-1, DL13, reference) | .00 | .99 | — | .363 | .464 |
| kl0099_trial90 (weak) | .071 | .000 | .915 | .369 | .481 |
| kl0253_trial42 (mid) | .398 | .004 | .592 | .368 | .483 |
| **kl0562_trial103 (strongest)** | **.883** | **.012** | .063 | .365 | .482 |

Monotonic and clean: as Heretic's KL budget rises, harm climbs smoothly (.07→.40→.88) while
ARC/MMLU stay flat at clean level (~.37/.48) the whole way and gibberish never rises above
1.2%. This is not the wall dissolving into noise (that pattern — high gib, some harm leak —
is what gemma v7 showed). This is a **direct, controllable trade of refusal for harm at zero
capability cost** — the exact failure mode "smart-and-safe XOR dumb-and-dangerous" is
supposed to prevent, fully realized.

**Trial103 fully breaks the wall.** 88.3% coherent, judged-harmful output at essentially
clean-level capability (ARC .365 / MMLU .482, vs clean baseline .384 / .484) — not a partial
leak, not gibberish-with-some-harm-mixed-in like gemma v7 ever showed under Heretic. Heretic's
TPE search found a per-layer/per-projection parameter combination (not a single scalar
strength) that the rank-1 attack never explores, and the training-time attack simulation
never covered either.

Compare to gemma v7 vs Heretic (`results/ext_heretic_out_v7_*`, different arch + product
version, same tool): worst tested case there was .171 harmAct / .662 gib (KL .27) — still
majority-gibberish. Llama v8's worst case is .883 harmAct / .012 gib — categorically
different outcome, essentially a clean jailbreak. Whether the gap is architecture (Llama
vs gemma) or product version (v7 vs v8) is confounded — no gemma-v8-vs-Heretic or
Llama-v7-vs-Heretic data exists yet to separate them.

**Implication:** ABL-v8's defense does not generalize to Heretic-style adaptive
per-projection ablation on Llama-3.2-1B. This is the single clearest adaptive-attack
failure found against v8 so far and needs to go in the paper's limitations section, not
get buried. Next question: does the same trial-103-style attack break gemma v8 / Qwen v8
too, or is this Llama-specific (its rank-1 wall was already the leakiest of the three,
per `docs/paper_tables_v1.md` Table 3).

## Backup

Pushed to private HF (`aaronrockmenezes/tamperforge`), `attacked_snapshots/heretic_llama_v8_{tag}/`
for all 3 trials, ~2.4GB each (7.2GB total). Uncensored weights — repo stays private.
- kl0562_trial103: https://huggingface.co/aaronrockmenezes/tamperforge/commit/063aa5b7991bae77e7792c6bed5c4ff2b17d976d
- kl0099_trial90: https://huggingface.co/aaronrockmenezes/tamperforge/commit/9ed393c782abf5fd65fea32fd1de17650c50510b
- kl0253_trial42: https://huggingface.co/aaronrockmenezes/tamperforge/commit/3b3af1608f3923d911ee2fe5c4f0c38580023a58

## Provenance

- Prior Heretic data (gemma v7 + base only): `results/ext_heretic_out_v7_*`,
  `results/heretic_base_gemma_*`, `results/cap_ext_heretic_out_v7_*`,
  `results/cap_heretic_base_*`.
- Attack zoo entry: `docs/attack_zoo_v0.md` Tier 2 #5 (Heretic/ARA).
- v8 rank-1 baseline numbers this run is compared against: `docs/paper_tables_v1.md`
  Tables 2–3.
