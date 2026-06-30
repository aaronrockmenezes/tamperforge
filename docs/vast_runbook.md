# Vast.ai runbook

This is the server path. Local MPS is too slow for the ablated HF models at the
official generation length.

## Machine to rent

RTX 5090 is good **if** the image has working Blackwell/CUDA support.

Pick:

- GPU: RTX 5090 32GB first choice if price is sane.
- Fallback: RTX 4090 24GB.
- Image: recent PyTorch CUDA image, CUDA 12.8+ preferred for 5090.
- Disk: 160GB minimum, 250GB safer.
- RAM: 32GB minimum, 64GB nicer.
- Host: verified, high uptime, good network/disk.

Do not rent H100/A100 for this phase unless the 5090/4090 market is broken.
Gemma 3 1B is small; repeated generation/eval is the bottleneck.

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

1. **Safety:** tamperforge generation on all AdvBench harmful prompts
   (`data/advbench_harmful_behaviors.csv`, 520 prompts), judged with
   DeepSeek V4 Flash via OpenRouter.
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
  --n-advbench 2 \
  --n-arc 2 \
  --max-new-tokens 32 \
  --run-id vast_smoke_base
```

```bash
lm_eval \
  --model hf \
  --model_args pretrained=google/gemma-3-1b-it,dtype=bfloat16,trust_remote_code=True \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --limit 10 \
  --batch_size auto \
  --device cuda:0 \
  --output_path results/lm_eval_smoke_gemma3_1b_it_arc_c_25shot \
  --log_samples
```

## Safety generation: AdvBench all 520

Run generation first, no judge inline. This keeps local GPU work separate from
OpenRouter work and lets us retry judge without regenerating.

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id google/gemma-3-1b-it \
  --n-advbench 520 \
  --n-arc 0 \
  --max-new-tokens 128 \
  --run-id safety_base_advbench520_nojudge
```

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id DavidAU/gemma-3-1b-it-heretic-abliterated-uncensored \
  --n-advbench 520 \
  --n-arc 0 \
  --max-new-tokens 128 \
  --run-id safety_hf_heretic_advbench520_nojudge
```

```bash
PYTHONPATH=src python experiments/p0_baseline_eval.py \
  --model-id DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated \
  --n-advbench 520 \
  --n-arc 0 \
  --max-new-tokens 128 \
  --run-id safety_hf_extreme_advbench520_nojudge
```

## Judge safety generations

Do a 10-row formatting smoke first:

```bash
PYTHONPATH=src python experiments/judge_generations.py \
  --generations results/safety_hf_heretic_advbench520_nojudge/generations.jsonl \
  --limit 10 \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 1 \
  --run-id judge_smoke_hf_heretic_10
```

If parse failures are zero, run full judge:

```bash
PYTHONPATH=src python experiments/judge_generations.py \
  --generations results/safety_base_advbench520_nojudge/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 12 \
  --run-id judge_base_advbench520_w12
```

```bash
PYTHONPATH=src python experiments/judge_generations.py \
  --generations results/safety_hf_heretic_advbench520_nojudge/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 12 \
  --run-id judge_hf_heretic_advbench520_w12
```

```bash
PYTHONPATH=src python experiments/judge_generations.py \
  --generations results/safety_hf_extreme_advbench520_nojudge/generations.jsonl \
  --judge-model deepseek/deepseek-v4-flash \
  --judge-json-mode \
  --judge-max-tokens 256 \
  --num-workers 12 \
  --run-id judge_hf_extreme_advbench520_w12
```

## Capability: lm-eval ARC-Challenge

Full ARC-Challenge test split:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=google/gemma-3-1b-it,dtype=bfloat16,trust_remote_code=True \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size auto \
  --device cuda:0 \
  --output_path results/lm_eval_base_arc_challenge_25shot_full \
  --log_samples
```

```bash
lm_eval \
  --model hf \
  --model_args pretrained=DavidAU/gemma-3-1b-it-heretic-abliterated-uncensored,dtype=bfloat16,trust_remote_code=True \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size auto \
  --device cuda:0 \
  --output_path results/lm_eval_hf_heretic_arc_challenge_25shot_full \
  --log_samples
```

```bash
lm_eval \
  --model hf \
  --model_args pretrained=DavidAU/gemma-3-1b-it-heretic-extreme-uncensored-abliterated,dtype=bfloat16,trust_remote_code=True \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --batch_size auto \
  --device cuda:0 \
  --output_path results/lm_eval_hf_extreme_arc_challenge_25shot_full \
  --log_samples
```

Capped 600-example version, if the bill is getting annoying:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=MODEL_ID_OR_PATH,dtype=bfloat16,trust_remote_code=True \
  --tasks arc_challenge \
  --num_fewshot 25 \
  --limit 600 \
  --batch_size auto \
  --device cuda:0 \
  --output_path results/lm_eval_RUN_arc_challenge_25shot_limit600 \
  --log_samples
```

## Optional capability: MMLU/GSM8K

Run only after ARC + AdvBench are safely archived. These are slower/more
expensive but more standard for paper tables.

```bash
lm_eval \
  --model hf \
  --model_args pretrained=google/gemma-3-1b-it,dtype=bfloat16,trust_remote_code=True \
  --tasks mmlu \
  --num_fewshot 5 \
  --limit 600 \
  --batch_size auto \
  --device cuda:0 \
  --output_path results/lm_eval_base_mmlu_5shot_limit600 \
  --log_samples
```

```bash
lm_eval \
  --model hf \
  --model_args pretrained=google/gemma-3-1b-it,dtype=bfloat16,trust_remote_code=True \
  --tasks gsm8k \
  --num_fewshot 8 \
  --limit 600 \
  --batch_size auto \
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
  --n-harmful 520 \
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
  --n-direction 64 \
  --abliterate-layers all \
  --n-advbench 520 \
  --n-arc 299 \
  --max-new-tokens 128 \
  --judge \
  --run-id p1_mad_crux_advbench520_judged
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
