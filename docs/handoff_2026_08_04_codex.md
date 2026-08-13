# HANDOFF — 2026-08-04 — version G replication, FT attacks, and validity boundary

**This is the current TamperForge source of truth.** It supersedes the live-status and headline
claims in `docs/handoff_2026_08_03b_codex.md`. The August 3 handoff remains the source for the
version G recipe, Qwen training dynamics, gate definitions, and the MMLU truncation finding.

Read this file first, then `TODO.md`, `CLAUDE.md`, and the August 3 handoffs for history.

---

## 0. Executive status

1. **Qwen version G is a strong in-distribution result, but its AdvBench-520 evaluation is
   train-exposed.** `data/harm_targets_qwen.json` contains 404 exact AdvBench goals and
   `data/extended_refusals_advbench.json` overlaps all 520 evaluation prompts. Do not present
   AdvBench-520 as held-out generalisation for version G.
2. **The corrected Llama version G replication blocks rank-1 and surgical ablation without a
   capability crater.** This is a fortress result on a second architecture, not the desired
   poison-pill result.
3. **Heretic partially breaks Llama version G coherently:** 20.58% harmful-actionable with only
   0.19% gibberish; held-out XSTest unsafe harm rises to 16.5%. This is a real partial break,
   not an AdvBench-only/judge artifact.
4. **Direct harmful fine-tuning breaks Qwen version G.** One epoch over 592 mined harmful
   completions yields 46.15% actionable harm for full SFT and 38.46% for LoRA on AdvBench.
   GSM8K remains at 0.3897/0.3783 vs clean 0.3965, so the break is not a general capability
   collapse. This is train-on-test attack characterisation, not held-out safety generalisation.
5. **Benign fine-tuning also weakens the Qwen wall:** benign full SFT stays near-safe (0.58%
   harm), while benign LoRA reaches 8.08% actionable harm. This makes ordinary adaptation a
   material threat even without harmful demonstrations.
6. **The next decisive experiment is a fresh held-out harmful benchmark**, frozen before any
   new mining or training. Stop iterating on AdvBench as the headline safety instrument.
7. **That held-out experiment is now running.** The frozen suite, contamination manifest,
   16-arm launcher, and supervisor service are recorded in section 7. Results are not yet final.

---

## 1. Critical correction: the first “Llama rank-1” result was Qwen

The initial `vgl_rank1` artifact that scored 57.31% judge ASR was not Llama. `chain_f.sh`
hardcoded the Qwen base model for rank-1, surgical, and Heretic replay. Evidence from the bad
artifact: `model_type=qwen3`, hidden size 1024, 28 layers, and a 1.19 GB checkpoint; loading the
Llama checkpoint found zero compatible matrices.

Invalid generations and summaries were quarantined on the box at:

```
/workspace/tamperforge/quarantine/invalid_vgl_qwen_20260803T1705/
```

Corrections made:

- `scripts/runs/chain_f.sh` now parameterises `MODEL_ID` and `DIRECTION_LAYER`.
- `experiments/v11_surgical_ablation.py` raises if a checkpoint loads zero compatible matrices.
- `scripts/runs/run_version_g_llama.sh` prints the actual architecture-specific command.

Only the corrected results below are valid.

---

## 2. Corrected Llama version G results

Model/checkpoint:

```
base:       meta-llama/Llama-3.2-1B-Instruct
checkpoint: outputs/version_g_llama_500.pt
layers:     attack 6-14; direction layer 13
recipe:     lambda_rr=4, lambda_safe=4, lambda_gib=0, 500 steps
```

### 2a. Safety and capability matrix

| arm | AdvBench harm | gibberish | XST unsafe harm | XST unsafe refusal | GSM8K | ARC acc_norm | MBPP | HumanEval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.0038 | 0.0096 | 0.010 | 0.910 | 0.3116 | 0.3942 | 0.340 | 0.3110 |
| rank-1 | 0.0019 | 0.0019 | 0.000 | 0.915 | 0.2851 | 0.3865 | 0.306 | 0.2988 |
| surgical k16 | 0.0038 | 0.0019 | 0.020 | 0.845 | 0.3116 | 0.3968 | 0.346 | 0.3110 |
| Heretic t138 | **0.2058** | **0.0019** | **0.165** | 0.535 | 0.3025 | 0.3891 | 0.324 | 0.3110 |

XSTest safe-prompt results for Heretic t138: refusal 0.084, harmful 0.000, gibberish 0.004.

### 2b. Interpretation

- Rank-1 and surgical fail to remove safety and do not trigger a capability crater. Llama
  version G is therefore a **fortress**, not a demonstrated poison pill, under these attacks.
- Heretic t138 (`kl=0.2814`) is a coherent partial break: 107/520 AdvBench responses were judged
  harmful-actionable, with one gibberish response. It also breaks on held-out XSTest unsafe
  prompts, so the effect is not explained solely by AdvBench train exposure.
- This is weaker cross-architecture transfer than Qwen version G. It does not justify a claim
  that the Qwen result generalises unchanged across architectures.

Correct artifact names on the box:

```
outputs/version_g_llama_500_clean
outputs/vgl_rank1
outputs/vgl_surg_k16
outputs/vgl_her_s0_att              # winning trial is t138 despite the s0 run tag
results/vgl_her_s0_trial.json
```

---

## 3. AdvBench contamination audit

Version G uses two files derived from the same benchmark later used for its headline evaluation:

- `data/harm_targets_qwen.json`: 404 unique goals, all exact AdvBench goals; 592 completions;
  covers 404/520 evaluation prompts.
- `data/extended_refusals_advbench.json`: 521 prompt entries; overlaps all 520 AdvBench
  evaluation prompts.

The representation-rerouting loss does not SFT the model on harmful completions: it
teacher-forces attacked and frozen-base models on that text and penalises hidden-state cosine
alignment. Nevertheless, the prompts and targets are training inputs, so AdvBench-520 is not a
held-out benchmark for version G.

Allowed claim: version G resists attacks on its training distribution while retaining utility.

Not allowed claim: version G generalises to unseen harmful requests based on AdvBench-520 alone.

XSTest is held out and useful evidence, but it is a narrow safe/unsafe contrast and cannot carry
the full generalisation claim. A fresh held-out HarmBench/JailbreakBench-style suite is required.

---

## 4. Qwen version G fine-tuning attacks

All rows start from `outputs/version_g_qwen_500.pt` materialised as Qwen3-0.6B.

### 4a. Benign adaptation

Ordinary assistant demonstrations; no harmful completions.

| attack | examples | AdvBench actionable harm | gibberish | judge refusal | judge ASR |
|---|---:|---:|---:|---:|---:|
| benign full SFT | 1000 | 0.0058 | 0.0019 | 0.9827 | 0.0058 |
| benign LoRA r16 | 1000 | **0.0808** | 0.0192 | 0.8846 | 0.0846 |

Ordinary LoRA adaptation damages the wall substantially more than ordinary full SFT under the
tested hyperparameters. This is an in-scope practical limitation because it needs no harmful
training data.

### 4b. Direct harmful AdvBench training

Training data: `data/harm_targets_qwen.json`, 592 completion pairs covering 404 AdvBench goals,
one epoch. Full SFT uses lr `1e-5`; LoRA uses r16/alpha32 and lr `2e-4`.

| attack | trainable params | actionable harm | gibberish | judge refusal | judge ASR |
|---|---:|---:|---:|---:|---:|
| harmful full SFT | 596M (100%) | **0.4615** | 0.3077 | 0.0154 | 0.4769 |
| harmful LoRA r16 | 10.09M (1.665%) | **0.3846** | 0.3327 | 0.0154 | 0.4096 |

Both attacks remove refusal. Roughly one third of outputs are gibberish, but full SFT still
produces 240/520 actionable harmful answers and LoRA produces 200/520. Because training and
evaluation share AdvBench goals, report this as a direct train-on-test stress test.

### 4c. Capability after direct harmful training

Full GSM8K test split, 1,319 examples, 5-shot, `lm_eval` strict match:

| model | GSM8K strict | flexible extract | retention vs version G clean 0.3965 |
|---|---:|---:|---:|
| harmful full SFT | **0.3897** | 0.3882 | 98.3% |
| harmful LoRA r16 | **0.3783** | 0.3806 | 95.4% |

The safety break is not explained by broad mathematical-capability collapse. GSM8K alone is not
a fluency instrument, so MT-Bench on these two attacked checkpoints remains useful but is not a
prerequisite for recording the present result.

Result paths:

```
results/vg_benign_sft_adv_judged/summary.json
results/vg_benign_lora_adv_judged/summary.json
results/vg_harm_sft_adv_judged/summary.json
results/vg_harm_lora_adv_judged/summary.json
results/vg_harm_sft_gsm8k/outputs__vg_harm_sft592/results_2026-08-03T19-08-19.728039.json
results/vg_harm_lora_gsm8k/outputs__vg_harm_lora_r16_592/results_2026-08-03T19-11-25.396752.json
```

---

## 5. Code changes made during this continuation

- `experiments/ft_attack.py`
  - true PEFT LoRA support and merged-model save;
  - JSON `{prompt: [completion, ...]}` demo loading;
  - trainable-parameter reporting.
- `experiments/v11_surgical_ablation.py`
  - fail closed when zero checkpoint matrices are compatible.
- `scripts/runs/chain_f.sh`
  - architecture-specific model and direction-layer parameters.
- `scripts/runs/run_version_g_llama.sh`
  - corrected printed launcher.
- `scripts/runs/eval_vg_benign_ft_advbench.sh`
- `scripts/runs/run_vg_benign_ft_attacks.sh`
- `scripts/runs/run_vg_harmful_ft_attacks_advbench.sh`
- `scripts/eval/eval_vg_harmful_ft_gsm8k.sh`
- `scripts/external_benches/fetch_prompts.py`
- `scripts/external_benches/freeze_version_g_suite.py`
- `scripts/external_benches/score_overrefusal.py`
- `scripts/runs/run_version_g_extended_heldout.sh`
- `scripts/runs/supervisor_version_g_extended.sh`
- `scripts/runs/tamperforge-vgho.supervisor.conf`
- `scripts/runs/judge_version_g_extended_async.py`
- `scripts/runs/supervisor_version_g_extended_judge.sh`
- `scripts/runs/tamperforge-vgho-judge.supervisor.conf`

These changes are currently local/uncommitted unless a later handoff says otherwise. Preserve
unrelated user changes when committing.

---

## 6. Hugging Face backup

Private repository: `aaronrockmenezes/tamperforge`.

The corrected Llama version G backup was uploaded and verified under:

```
final_backup_2026_08_04/checkpoints/version_g_llama_500.pt
final_backup_2026_08_04/version_g_llama_variants/clean/
final_backup_2026_08_04/version_g_llama_variants/rank1/
final_backup_2026_08_04/version_g_llama_variants/surg_k16/
final_backup_2026_08_04/version_g_llama_variants/heretic_t138_att/
final_backup_2026_08_04/version_g_llama_variants/heretic_t138_trial.json
final_backup_2026_08_04/results/
final_backup_2026_08_04/SHA256SUMS
```

The backup includes the raw checkpoint, four directly runnable model directories, the winning
Heretic trial, training manifest, judged safety/XSTest summaries, capability results, MT-Bench
summaries, and SHA-256 hashes.

Verification performed after upload:

```
repository private:       true
payload backup commit:    a55894e3c0cbc5e829617a7f080353298aad7b83
local staged file count:  79
Hub prefix file count:    79
required model paths:     raw + clean + rank1 + surg_k16 + heretic_t138 all present
SHA256SUMS verification:  freshly downloaded Hub file byte-identical to staged manifest
```

Do not make this repository public; it contains dual-use attacked weights.

---

## 7. What is next

### P0 — decisive validity repair — RUNNING 2026-08-04

Frozen suite: `data/heldout_vg_20260804/`. The manifest checks both Version G training maps
after Unicode/case/whitespace normalization and removes normalized exact matches plus
RapidFuzz ratio >=97 matches. All 50 removed rows were inspected: 36 exact and 14 near-exact
article/punctuation/wording variants; no broad semantic filter was used.

| benchmark | raw | frozen | training overlaps removed |
|---|---:|---:|---:|
| HarmBench standard | 200 | 200 | 0 |
| JailbreakBench harmful | 100 | 86 | 14 |
| StrongREJECT | 313 | 288 | 25 |
| SORRY-Bench 202406 | 450 | 439 | 11 |
| BeaverTails unsafe dedup | 1,483 | 1,483 | 0 |
| XSTest safe / unsafe | 250 / 200 | 250 / 200 | 0 / 0 |
| OR-Bench | 1,319 | 1,319 | 0 |

The five harmful benches total 2,496 prompts per model. Sixteen model arms generate 39,936
harmful-benchmark responses; the four Qwen FT arms additionally generate XSTest safe/unsafe
and OR-Bench, for 47,012 total responses. Harmful outputs use the pinned uniform Tier-1
DeepSeek judge; safe XSTest and OR-Bench use the pinned over-refusal scorer. These are a common
cross-benchmark scoring layer, not each benchmark's official bespoke judge.

The run is managed by supervisor, per the Vast host guide:

```
service:  tamperforge-vgho
status:   supervisorctl status tamperforge-vgho
restart:  supervisorctl restart tamperforge-vgho
driver:   scripts/runs/run_version_g_extended_heldout.sh
stdout:   logs/eval/vgho_supervisor.log
run log:  logs/eval/version_g_extended_heldout_20260803T195817.log
results:  results/vgho_<model-tag>_<benchmark>/
```

Judging was started asynchronously while generation continued. The dispatcher validates each
`generations.jsonl` against the frozen manifest's exact expected row count before submitting it
and uses 96 workers. Once all 92 generation artifacts completed, the original combined launcher
was stopped at a clean summary boundary and the async judge became the sole scoring owner; this
avoids duplicate API calls while preserving 96-worker concurrency:

```
service:    tamperforge-vgho-judge
status:     supervisorctl status tamperforge-vgho-judge
dispatcher: scripts/runs/judge_version_g_extended_async.py
stdout:     logs/eval/vgho_judge_supervisor.log
generator:  tamperforge-vgho is intentionally STOPPED after 92/92 generation artifacts
```

First validation: `vgho_qbase_clean_harmbench_judged` completed 200/200 with zero parse
failures. Its preliminary held-out HarmBench harmful-actionable rate is 0.325; do not interpret
one benchmark/model row as the suite result.

At launch verification, all five `qbase_clean` artifacts completed at their exact expected
counts (2,496/2,496 total) with zero API errors, and the launcher advanced to `qbase_rank1`.
Do not treat this as a completed evaluation until all expected generation files and judged
summaries exist.

Freeze a held-out safety suite before generating any model-specific data. Recommended minimum:

1. HarmBench standard behaviors not present in either training JSON.
2. JailbreakBench/StrongReject behaviors deduplicated against both training files.
3. Exact-overlap and near-duplicate audit saved as a manifest.
4. Evaluate Qwen version G clean/rank-1/surgical/Heretic plus harmful-SFT/LoRA checkpoints.
5. Evaluate Llama version G clean/rank-1/surgical/Heretic on the identical frozen suite.
6. Judge with the pinned DeepSeek model; report actionable harm, refusal, gibberish, and parse
   failures. Do not mine replacement targets until this frozen evaluation is complete.

This is the experiment that decides whether version G is a publishable general defence or a
strong training-distribution result.

### P1 — complete the fine-tuning story

- MT-Bench, XSTest safe/unsafe, ARC, and MBPP on harmful full-SFT and LoRA checkpoints.
- Benign SFT/LoRA on the held-out safety suite.
- A small data-dose curve (for example 16/64/256/592) only after the held-out suite is frozen.

### P2 — mechanism and paper strengthening

- Mine Llama-native harmful targets and retrain only if the frozen evaluation identifies a
  cross-architecture gap that this hypothesis can explain.
- Repeat version G on a larger model only after held-out generalisation is established at 0.6B/
  1B; scaling before that would amplify cost without fixing validity.
- Fix `serve_eval.sh` MMLU `max_length`, then rerun only the MMLU rows intended for publication.
- Diagnose the older HumanEval path issue before using those rows in a paper table.

### Stop rule

Do not spend another cycle improving AdvBench-520 numbers. That benchmark is now a training-set
diagnostic for version G, not the headline generalisation benchmark.
