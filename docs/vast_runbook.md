# Vast.ai runbook

This is the server path. Local MPS is too slow for the ablated HF models at the
official generation length.

## Machine to rent

RTX 5090 is good **only if the host driver exposes CUDA 12.9+**. A 5090 host
showing `CUDA Version: 12.8` in `nvidia-smi` is not usable for our vLLM path:
vLLM/FlashAttention fails on Blackwell SM 12.x with driver/runtime mismatch.

Pick:

- GPU: RTX 4090 24GB is the safest cheap choice.
- Alternate: RTX 5090 32GB only if `nvidia-smi` reports CUDA 12.9+.
- Fallback: L40S/A100/H100 if 4090 market is bad.
- Image: recent PyTorch CUDA image.
- Disk: 160GB minimum, 250GB safer.
- RAM: 32GB minimum, 64GB nicer.
- Host: verified, high uptime, good network/disk.

Do not rent H100/A100 for this phase unless the 4090 market is broken.
Gemma 3 1B is small; repeated generation/eval is the bottleneck.

Known bad box:

```text
NVIDIA GeForce RTX 5090, Driver 570.195.03, CUDA Version 12.8
```

This imports vLLM after CUDA library path repair, but generation fails during
engine startup with `CUDA driver version is insufficient for CUDA runtime
version`. Stop that instance instead of debugging further.

## Model list

Pinned current HF revisions checked 2026-07-01:

| label | model/path | revision |
|---|---|---|
| base | `google/gemma-3-1b-it` | `dcc83ea841ab6100d6b47a070329e1ba4cf78752` |
| hf_heretic | `DavidAU/gemma-3-1b-it-heretic-abliterated-uncensored` | `e97a6914cbf98f9a7584e5dad9e53aaf76d86d72` |
| hf_extreme | `DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated` | `5a2a35d933daecb71f5f71ab0b4661b60e73ccd3` |
| local_all_empirical | `outputs/gemma3_1b_it_abliterated_all_empirical` | generated on server |
| adapter_p1 | `outputs/safety_adapter_p1.pt` | trained on server |

Local L13/all-layer SAE checkpoints are legacy comparisons. Do not spend server
time on them before the HF baselines and empirical all-layer path.

## Benchmark plan

Use two tracks:

1. **Safety:** tamperforge generation on HF `walledai/AdvBench`, capped to 500
   prompts for the official run, judged with DeepSeek V4 Flash via OpenRouter.
   The vendored local CSV is legacy/internal only.
2. **Capability:** EleutherAI `lm-eval` for standard capability reporting.
   Primary: ARC-Challenge. Full ARC-Challenge test split is 1,172 examples;
   validation split is 299. If time is tight, use `--limit 600`, but label it
   as a capped run. Full run is preferred.

Our custom ARC loop is useful for quick internal checks, but publication-facing
capability numbers should come from `lm-eval`.

Gemma model-card caveat: Google's card reports Gemma 3 **PT** 1B ARC-c 25-shot
as 38.4. It does not give a clean official `gemma-3-1b-it` lm-eval ARC number.
So the server gate is: first run `google/gemma-3-1b-it` through the exact
`lm-eval` command below, archive that as the local reference, then compare every
ablated/adapted model against that same command and model revision.

## Setup

Use the active Vast environment. On the current instance that is:

```bash
conda env list
# main * /venv/main
```

Do not create a new env unless the image is broken; `scripts/vast_setup.sh`
uses `python` from the active env by default.

Observed good RTX 4090 package state:

```text
Python 3.12.13
torch 2.11.0+cu130
torch.version.cuda 13.0
vLLM 0.24.0
GPU NVIDIA GeForce RTX 4090
```

This is valid even if an intermediate install mentions CUDA 12.8 wheels. The
driver can run older bundled CUDA runtimes.

```bash
nvidia-smi
python --version
git clone git@github.com:aaronrockmenezes/tamperforge.git
cd tamperforge
bash scripts/vast_setup.sh
```

If SSH is annoying:

```bash
git clone https://github.com/aaronrockmenezes/tamperforge.git
cd tamperforge
bash scripts/vast_setup.sh
```

If Torch CUDA is missing on the image:

```bash
INSTALL_TORCH=1 bash scripts/vast_setup.sh
```

Add OpenRouter key before judge runs:

```bash
printf "OPENROUTER_API_KEY=%s\n" "YOUR_KEY" > .env
chmod 600 .env
```

## Smoke

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id google/gemma-3-1b-it \
  --backend vllm \
  --n-advbench 2 \
  --n-arc 0 \
  --max-new-tokens 32 \
  --max-length 4096 \
  --vllm-batch-size 2 \
  --run-id vast_smoke_base
```

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=google/gemma-3-1b-it,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --limit 10 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path results/lm_eval_smoke_gemma3_1b_it_arc_c_25shot \
  --log_samples
```

## Safety generation: AdvBench 500

Run generation first, no judge inline. This keeps local GPU work separate from
OpenRouter work and lets us retry judge without regenerating.

Safety generation uses vLLM batching. `generations.jsonl` is appended after each
completed vLLM batch.

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id google/gemma-3-1b-it \
  --backend vllm \
  --advbench-source walledai \
  --n-advbench 500 \
  --n-arc 0 \
  --max-new-tokens 512 \
  --max-length 4096 \
  --vllm-batch-size 64 \
  --vllm-gpu-memory-utilization 0.9 \
  --run-id safety_base_advbench500_nojudge
```

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id DavidAU/gemma-3-1b-it-heretic-abliterated-uncensored \
  --backend vllm \
  --advbench-source walledai \
  --n-advbench 500 \
  --n-arc 0 \
  --max-new-tokens 512 \
  --max-length 4096 \
  --vllm-batch-size 64 \
  --vllm-gpu-memory-utilization 0.9 \
  --run-id safety_hf_heretic_advbench500_nojudge
```

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated \
  --backend vllm \
  --advbench-source walledai \
  --n-advbench 500 \
  --n-arc 0 \
  --max-new-tokens 512 \
  --max-length 4096 \
  --vllm-batch-size 64 \
  --vllm-gpu-memory-utilization 0.9 \
  --run-id safety_hf_extreme_advbench500_nojudge
```

## Judge safety generations

Do a 10-row formatting smoke first:

```bash
PYTHONPATH=src python experiments/judge_generations.py \
  --generations results/safety_hf_heretic_advbench500_nojudge/generations.jsonl \
  --limit 10 \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 1 \
  --run-id judge_smoke_hf_heretic_10_v2
```

If parse failures are zero, run full judge. The current judge client disables
OpenRouter reasoning; without that, DeepSeek can return `content: null` after
spending the whole completion budget on hidden reasoning.

```bash
PYTHONPATH=src python experiments/judge_generations.py \
  --generations results/safety_base_advbench500_nojudge/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 12 \
  --run-id judge_base_advbench500_w12
```

```bash
PYTHONPATH=src python experiments/judge_generations.py \
  --generations results/safety_hf_heretic_advbench500_nojudge/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 12 \
  --run-id judge_hf_heretic_advbench500_w12
```

```bash
PYTHONPATH=src python experiments/judge_generations.py \
  --generations results/safety_hf_extreme_advbench500_nojudge/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 12 \
  --run-id judge_hf_extreme_advbench500_w12
```

## Capability: lm-eval ARC-Challenge

ARC-Challenge uses loglikelihood scoring. On 24GB GPUs, do not let vLLM default
to Gemma's 32k context or `--batch_size auto`; that can OOM while computing
prompt logprobs. Keep the command identical across models.

Full ARC-Challenge test split:

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

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=DavidAU/gemma-3-1b-it-heretic-abliterated-uncensored,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path results/lm_eval_hf_heretic_arc_challenge_25shot_full \
  --log_samples
```

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path results/lm_eval_hf_extreme_arc_challenge_25shot_full \
  --log_samples
```

Capped 600-example version, if the bill is getting annoying:

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=MODEL_ID_OR_PATH,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --limit 600 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path results/lm_eval_RUN_arc_challenge_25shot_limit600 \
  --log_samples
```

## Optional capability: MMLU/GSM8K

Run only after ARC + AdvBench are safely archived. These are slower/more
expensive but more standard for paper tables.

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=google/gemma-3-1b-it,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
  --tasks mmlu \
  --num_fewshot 5 \
  --limit 600 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path results/lm_eval_base_mmlu_5shot_limit600 \
  --log_samples
```

```bash
lm_eval \
  --model vllm \
  --model_args pretrained=google/gemma-3-1b-it,dtype=bfloat16,trust_remote_code=True,max_model_len=4096,gpu_memory_utilization=0.65,max_num_seqs=1 \
  --tasks gsm8k \
  --num_fewshot 8 \
  --limit 600 \
  --batch_size 1 \
  --device cuda:0 \
  --output_path results/lm_eval_base_gsm8k_8shot_limit600 \
  --log_samples
```

## Create local empirical ablated model

Do this after the HF baselines unless there is time to spare.

```bash
PYTHONPATH=src python experiments/make_abliterated_model.py \
  --model-id google/gemma-3-1b-it \
  --out outputs/gemma3_1b_it_abliterated_all_empirical \
  --advbench-source walledai \
  --direction-source empirical \
  --direction-layer 13 \
  --n-direction 64 \
  --layers all
```

Then run the same safety and lm-eval commands with:

```bash
--model-id outputs/gemma3_1b_it_abliterated_all_empirical
```

and:

```bash
--model_args pretrained=outputs/gemma3_1b_it_abliterated_all_empirical,dtype=bfloat16,trust_remote_code=True
```

## Train P1 adapter

Fast first pass:

```bash
PYTHONPATH=src python experiments/train_adapter.py \
  --model-id google/gemma-3-1b-it \
  --out outputs/safety_adapter_p1.pt \
  --advbench-source walledai \
  --n-harmful 200 \
  --epochs 5 \
  --batch-size 4 \
  --lr 3e-4 \
  --lambda-entangle 1.0 \
  --lambda-suppress 0.5 \
  --run-id train_adapter_p1_fast
```

Stronger if time remains:

```bash
PYTHONPATH=src python experiments/train_adapter.py \
  --model-id google/gemma-3-1b-it \
  --out outputs/safety_adapter_p1_full.pt \
  --advbench-source walledai \
  --n-harmful 500 \
  --epochs 20 \
  --batch-size 4 \
  --lr 3e-4 \
  --lambda-entangle 1.0 \
  --lambda-suppress 0.5 \
  --run-id train_adapter_p1_full
```

## Run fixed P1 POC

```bash
PYTHONPATH=src python experiments/p1_mad_crux.py \
  --model-id google/gemma-3-1b-it \
  --adapter outputs/safety_adapter_p1.pt \
  --advbench-source walledai \
  --n-direction 64 \
  --abliterate-layers all \
  --n-advbench 500 \
  --n-arc 299 \
  --max-new-tokens 512 \
  --max-length 4096 \
  --judge \
  --run-id p1_mad_crux_advbench500_judged
```

P1 uses DeepSeek V4 Flash by default. It loads up to five conditions, so run it
after the standalone baselines unless budget/time is comfortable. P1 still uses
the internal ARC validation loop because the prototype adapter is a forward hook
and is not exposed to `lm-eval` yet.

## Pull results back

```bash
tar -czf tamperforge_results_$(date +%Y%m%d_%H%M%S).tgz results outputs
```

Download before destroying the instance.
