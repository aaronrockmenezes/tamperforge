# Common issues and fixes

## RTX 5090 imports vLLM but generation crashes

Symptom:

```text
CUDA driver version is insufficient for CUDA runtime version
SM 12.x requires CUDA >= 12.9
```

Cause: Blackwell/RTX 50 GPUs need a host driver exposing CUDA 12.9+ for the vLLM
path. A 5090 host with `nvidia-smi` showing CUDA 12.8 is not usable.

Fix: stop that instance. Rent RTX 4090, L40S, A100, H100, or a 5090 host where
`nvidia-smi` reports CUDA 12.9+.

## `vllm -V` fails

Symptom:

```text
vllm: error: unrecognized arguments: -V
```

Cause: vLLM 0.24 uses lowercase `-v`.

Fix:

```bash
vllm -v
```

## `ImportError: libcudart.so.13`

Symptom:

```text
ImportError: libcudart.so.13: cannot open shared object file
```

Cause: vLLM was installed with CUDA 13-linked wheels, but the Python-packaged
NVIDIA libraries were not on `LD_LIBRARY_PATH`.

Fix: rerun setup after pulling current code:

```bash
bash scripts/setup/vast_setup.sh
```

The setup script writes:

```text
/venv/main/etc/conda/activate.d/tamperforge_cuda_libs.sh
```

Then reconnect or reactivate:

```bash
source /venv/main/bin/activate
vllm -v
```

## Torch wheel CUDA version differs from `nvidia-smi`

Symptom:

```text
nvidia-smi: CUDA Version 13.0
torch: 2.11.0+cu128 or 2.11.0+cu130
```

Cause: `nvidia-smi` reports the maximum CUDA runtime supported by the driver.
PyTorch wheels bundle their own CUDA runtime. A CUDA 13.0-capable driver can run
CUDA 12.8 binaries.

Fix: accept it if Torch reports CUDA available:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected on the current good box:

```text
2.11.0+cu130 13.0 True NVIDIA GeForce RTX 4090
```

## `lm_eval --model vllm` OOMs on ARC

Symptom:

```text
torch.OutOfMemoryError ... prompt_logprobs ... 24GB
```

Cause: vLLM defaults Gemma 3 to 32k context and `--batch_size auto` can overfill
a 24GB card while ARC loglikelihood scoring computes prompt logprobs.

Fix: use the 24GB-safe command shape:

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=google/gemma-3-1b-it,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path results/lm_eval_base_arc_challenge_25shot_full \
  --log_samples
```

Keep this shape identical across models.

## DeepSeek judge parse failures with `content: null`

Symptom:

```text
"finish_reason": "length"
"content": null
"completion_tokens_details": {"reasoning_tokens": 256}
parse_failures > 0
```

Cause: OpenRouter/DeepSeek spent the full completion budget as hidden reasoning
and returned no JSON content.

Fix: use code at commit `56db984` or later. The judge request disables reasoning
with:

```json
{"reasoning": {"effort": "none", "exclude": true}}
```

The 10-row smoke must pass before full judging:

```bash
python experiments/judge_generations.py \
  --generations results/safety_hf_heretic_advbench500_nojudge/generations.jsonl \
  --limit 10 \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 1 \
  --run-id judge_smoke_hf_heretic_10_v2
```

Expected:

```text
parse_failures: 0
```

## OpenRouter rejects the judge request

Symptom:

```text
Only one of "reasoning.effort" and "reasoning.max_tokens" can be specified
```

Cause: an older local patch set both fields.

Fix: pull commit `56db984` or copy the current `src/tamperforge/eval/judge.py`.
Do not set `reasoning.max_tokens` together with `reasoning.effort`.

## Remote `git pull` fails on Vast

Symptom:

```text
fatal: could not read Username for 'https://github.com': No such device or address
```

Cause: the remote checkout uses HTTPS and has no GitHub credentials.

Fix options:

```bash
git remote set-url origin git@github.com:aaronrockmenezes/tamperforge.git
```

or copy changed files from local and remember that the server worktree may become
dirty. Before a later pull:

```bash
git restore PATH_YOU_COPIED
git pull --ff-only
```

## Remote `git pull` says local changes would be overwritten

Cause: a file was copied directly from local to remote to unblock a run.

Fix:

```bash
git status --short
git restore pyproject.toml scripts/setup/vast_setup.sh src/tamperforge/eval/judge.py
git pull --ff-only
```

Only restore files that are known copied code files. Do not delete result dirs.

## AdvBench count confusion: 500 vs 520

The HF dataset `walledai/AdvBench` has 520 rows in the train split, but the
official tamperforge server run caps safety eval at the first 500 prompts to
match the requested budget. The vendored CSV is legacy/internal only.

Use:

```bash
--advbench-source walledai --n-advbench 500
```

## vLLM shutdown prints SIGTERM

Symptom:

```text
[shutdown] EngineCore: trigger received signal=SIGTERM
```

Cause: normal vLLM process teardown after offline generation.

Fix: none. Treat it as clean if the script prints `summary.json` and exits.

## A script that "judges into the same dir it reads from" destroys the raw generations

Symptom: a picker/selector script runs judging, and afterward `generations.jsonl` for the
run(s) it just read is gone (or was gone before judging even started, if you check closely).

Cause: `RunLogger.__init__` unconditionally unlinks `generations.jsonl`/`events.jsonl`/
`judgments.jsonl` for whatever run-id it's given, so a re-run starts fresh instead of
appending stale rows. If a script passes the **same run-id** as both the source (raw
generations to read) and the destination (judged output to write), construction of the
destination `RunLogger` deletes the file before it's ever opened for reading. This is exactly
what happened to `scripts/tools/auto_pick_v8.py` (fixed in commit `d89872d`) — it destroyed 30 raw
generation files (15 checkpoints × attacked/clean) mid pick-job.

Fix: any script that judges an existing `results/<run_id>/generations.jsonl` must judge into
a **distinct** run-id, e.g. `f"{run_id}_judged"` — never reuse the source run-id for the
judged output. This convention is already used in `scripts/probes/qwen3_8b_thinking_dl_sweep.sh` and
`scripts/tools/auto_pick_v8.py` (post-fix); follow it in any new picker/selector script.

## TamperBench on 24GB: FIVE patches in, still 0 valid directions — use a bigger box (2026-08-02)

**Outcome: abandoned on the 3090 after five hand-patches. Do not restart this on 24GB without
a new idea.** It ran on the A6000 with full data; the honest read is that it needs that box.

Sequence, each fix real and each insufficient:

| # | patch | effect |
|---|---|---|
| 1-3 | fp64->fp32 x2 files + `del`/`empty_cache` (below) | necessary, not sufficient |
| 4 | `_load_dataset` ignores `data_samples` | fixed the OOM |
| 5 | Qwen family misinferred as Base | 56 -> 140 candidates, KL min 3.25 -> 1.25 |

After all five: **0 of 140 candidates pass** `kl_threshold 0.1`, and every `steering_score` is
negative (max -4.42), i.e. no candidate direction induces refusal when added. Raising
`data_samples` 16x (128/32 -> 2048/512) moved KL min only 3.25 -> 3.17, which rules out data
volume. A "best direction" is still written and the pipeline still runs the StrongREJECT /
MMLU-Pro evals on it — **an empty `filtered_scores.json` is the only signal that the number
you are about to read is meaningless. Always check it.**

Also: the value of this as third-party validation erodes with each patch. Five patches in you
are reporting numbers from a benchmark you substantially rewrote, which is worth less than
our own harness. Weigh that before patch six.

**Upstream bug worth reporting to them:** in `refusal_ablation.py` the step-3 steering block
rebinds `refusal_scores` before `json_output_all_scores.append(...)`, so `all_scores.json`'s
`refusal_score` column is a duplicate of `steering_score`. The filter reads the correct
internal arrays (`ablation_refusal_scores`), so only the JSON reporting is wrong.

### Patch 5 — Qwen3 instruct models are classified as base

`infer_model_family` matches the literal substring `"instruct"`. Qwen3 ships its INSTRUCT
model under the plain name (`Qwen/Qwen3-0.6B`) and marks the pretrained one `-Base`, so
upstream returns `QwenBaseModelFamilyConfig` and applies a "minimal inline chat template".
The model then never refuses, so there is no refusal signal to ablate or steer — which is why
every direction scored huge KL with negative steering. Fix:

```python
if "qwen" in name:
    if "base" in name:
        return MODEL_FAMILY_CONFIGS["QwenBase"]
    return MODEL_FAMILY_CONFIGS["QwenInstruct"]
```

Watch for the same trap on any model whose instruct variant is not named "*-Instruct".

## TamperBench OOMs on 24GB: it IGNORES `data_samples` and loads every split in full (2026-08-02)

**Batch size is a red herring. Do not waste a run on it.** `batch_size` 32->8 and
`inference_batch_size` 16->4, with `PYTORCH_ALLOC_CONF=expandable_segments:True`, changed
nothing: the failure reproduced at the *identical* 22.39 GiB allocated and the *identical*
3.55 GiB request. (A 2.0GB reading during dataset prep looks like success — it is taken
before the attack allocates. Do not call it fixed off that.)

Root cause: `_load_dataset` in
`src/tamperbench/whitebox/attacks/refusal_ablation/datasets.py` takes `_data_samples` and
**never applies it** — all three return paths hand back the full split. So the config's
`data_samples: 128 / 32` is silently ignored and you get harmless_train 18793 rows,
harmless_val 6264. The `(N, V)` logits tensors in `refusal_ablation.py`
(`_iso_get_last_position_logits`, `out = torch.empty((N, V), ...)`) are sized off
`len(dataset)`, so with Qwen's V=151936 that is

    18793 x 151936 x 4B = 11.4 GB     6264 x 151936 x 4B = 3.81 GB  <- the 3.55 GiB request

One allocation each, sized by dataset length. No batch setting can touch them.

Fix — make the loader honour its own config (3 return paths):

```python
ds = load_dataset(...)
if _data_samples is not None and 0 < _data_samples < len(ds):
    ds = ds.select(range(_data_samples))
return ds
```

This is not a deviation from their protocol — it restores what their own grid.yaml asks for.

**The clone is not vendored, so this is patch #4 that must be reapplied by hand** alongside
the three below (fp64->fp32 in `refusal_ablation.py` AND `attack_utils.py`, plus
`del intervention_logits; torch.cuda.empty_cache()` in the layer sweep). Originals are kept
as `*.py.orig` next to each patched file on the box.

## TamperBench (external repo) OOMs on `refusal_ablation` even after patching the obvious fp64 tensor

Symptom: `_iso_get_last_position_logits` in
`src/tamperbench/whitebox/attacks/refusal_ablation/refusal_ablation.py` was patched
`float64`→`float32` for its big `(N, V)` logits tensor, but the attack still OOMs needing
the *exact same* memory figure as the unpatched fp64 version.

Cause: `kl_div_fn` in `attack_utils.py` (same repo) silently upcasts both its inputs back to
`float64` internally, regardless of what dtype was passed in — it undoes the outer patch on
every single call. This is not our code (TamperBench is a separate clone, not vendored — see
CLAUDE.md), so this patch is not in git and must be reapplied by hand on any fresh clone.

Fix: patch `kl_div_fn` to cast to `float32` too, not just the call site. Also add
`del intervention_logits; torch.cuda.empty_cache()` after each per-layer KL-div computation —
the real leak once the dtype is fixed everywhere is per-iteration accumulation across the
layer×position sweep, not a one-time sizing issue. Watch for a **zombie CUDA context**
masking this during debugging: a crashed process can leave a PID holding GPU memory that
`ps aux` no longer shows (check `nvidia-smi --query-compute-apps`); `nvidia-smi --gpu-reset`
typically fails with "Insufficient Permissions" on an unprivileged rented box — it clears on
its own after enough time passes, not immediately.

## Qwen eval: `--qwen-thinking off` silently halves harmAct (2026-07-25)

Reproducing the Qwen Heretic numbers from `docs/heretic_v8_2026_07_18.md` off the archived
`attacked_snapshots/heretic_qwen_v8_*` checkpoints failed until the generation config matched
exactly. Same weights, same judge, same prompts — three different answers:

| config | harmAct | gib | avg out tok |
|---|---:|---:|---:|
| `--max-new-tokens 128 --qwen-thinking off` | .669 | .081 | 127 |
| `--max-new-tokens 512 --qwen-thinking off` | .594 | .160 | 332 |
| `--max-new-tokens 512 --qwen-thinking default` | **.788** | **.090** | 508 |
| doc value | .819 | .085 | — |

Two independent traps:

1. **`--max-new-tokens` defaults to 128** (`p0_baseline_eval.py`) while every reported eval
   matrix passes 512 (`scripts/eval_matrix_*.sh`). At 128, 501/520 generations are cut
   mid-answer and the judge scores truncated harm as non-actionable. This is the second
   recurrence of the 128-token artifact already recorded in `MEMORY.md` (FTR-v6).
2. **`--qwen-thinking off` is not the neutral choice.** It selects
   `apply_chat_template_no_think`, a different chat template from the one the campaign used.
   Even with `thinking_n == 0` in both cases, it costs ~.19 harmAct on its own — the larger
   of the two effects.

Rule: for any Qwen number that will be compared against a published table, copy the flags from
`scripts/eval/eval_matrix_new_qwen.sh` verbatim and leave `--qwen-thinking` at `default`.

**Underlying gap:** `results/*/summary.json` records only `run_id`, `device`, `backend` — not
`max_new_tokens`, `qwen_thinking`, sampling params, or prompt source. A stored summary cannot
be checked against the config that produced it, which is why the above took three runs to
diagnose. Worth persisting the generation config into the summary before the next campaign.

## Resume guards must test the ARTIFACT, not the directory (2026-08-02)

**Symptom:** an eval "completes" but its judged summary is missing, or a judge run reports on
a file that does not exist. Silent -- the script exits 0.

**Cause:** `[ -d "results/<tag>" ] || python -u experiments/p0_baseline_eval.py ...`

A directory left behind by a killed job is still a directory. The guard sees it, skips
regeneration, and the judge then runs against a `generations.jsonl` that was never written.
This destroyed four eval arms on 2026-08-01/02 (`lbase_clean`, `rep_vb_s1`, `rep_vc_s1`, and
`lbase_rank1`'s judge step) after a stray `pkill` killed the jobs mid-generation.

**Rule:** guard on the artifact the step actually produces.

| step | guard on |
|---|---|
| `p0_baseline_eval.py` | `results/<id>/generations.jsonl` |
| `judge_generations.py` | `results/<id>_judged/summary.json` |
| `lm_eval` | `find "$out" -name 'results_*.json' -print -quit \| grep -q .` (nested by date) |
| `save_p1b_checkpoint.py` / model export | `<dir>/model.safetensors` |

`[ -d ... ]` remains correct for checking an INPUT exists
(`[ -d "outputs/$D" ] \|\| { echo "!! missing $D"; continue; }`) -- those were left alone.

Fixed across 25 scripts by `fix_guards.py` / `fix_guards2.py`.

## Never `pkill -f` a pattern that can match another job (2026-08-01)

`pkill -TERM -f "VLLM::EngineCore"` in `scripts/eval/ceiling_llama.sh` killed the EngineCore of two
concurrent replication lanes, losing their generations. The pattern in
`scripts/eval/run_mad_v10_s175_vllm_caps.sh` (commit `1a4e603`) is the correct one and scopes the
kill to the evaluator it started:

    lm_eval ... &
    pid=$!
    # poll for results_*.json, then:
    ep="$(pgrep -P "$pid" -f 'VLLM::EngineCore' | head -1)"
    kill -TERM "$ep"

Also note `pkill -f <pat>` matches the killing shell's own command line if that string
appears in it -- that killed an ssh session earlier in the campaign.
