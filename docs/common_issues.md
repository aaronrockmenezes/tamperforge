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
bash scripts/vast_setup.sh
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
git restore pyproject.toml scripts/vast_setup.sh src/tamperforge/eval/judge.py
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
what happened to `scripts/auto_pick_v8.py` (fixed in commit `d89872d`) — it destroyed 30 raw
generation files (15 checkpoints × attacked/clean) mid pick-job.

Fix: any script that judges an existing `results/<run_id>/generations.jsonl` must judge into
a **distinct** run-id, e.g. `f"{run_id}_judged"` — never reuse the source run-id for the
judged output. This convention is already used in `scripts/qwen3_8b_thinking_dl_sweep.sh` and
`scripts/auto_pick_v8.py` (post-fix); follow it in any new picker/selector script.

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
