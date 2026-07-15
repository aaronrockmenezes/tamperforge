# Qwen3-8B Thinking-Mode ABL-v8 on RTX PRO 6000

Target box: 1x RTX PRO 6000 Blackwell, 96GB VRAM, at least 1TB free NVMe.

Protocol decision: this run is explicitly Qwen3 thinking-on. Training,
direction estimation, vLLM safety generation, snapshot picking, and four-cell
safety evals all pass thinking mode deliberately. Existing historical runs stay
no-thinking by default.

## Source facts

- Qwen3-8B is 8.2B params, 36 layers, Apache-2.0.
- Qwen3 tokenizer defaults to thinking mode, and accepts
  `enable_thinking=True` / `False` in `apply_chat_template`.
- Qwen recommends thinking-mode sampling with temperature 0.6, top-p 0.95,
  top-k 20, and warns against greedy decoding for thinking mode.
- vLLM supports Qwen3 reasoning and `chat_template_kwargs`; Qwen docs recommend
  `vllm>=0.8.5`, and vLLM docs list `qwen3` as a reasoning parser.

## 0. Setup

```bash
cd /workspace/tamperforge
bash scripts/setup_blackwell_pro6000.sh
```

This verifies the GPU, runs the standard Vast setup, checks the Qwen3 thinking
chat template, then loads Qwen3-8B once through vLLM.

## 1. Direction-layer sweep

```bash
cd /workspace/tamperforge
DLS="12 16 20 24 28 32" JUDGE=1 bash scripts/qwen3_8b_thinking_dl_sweep.sh
```

Pick the layer from:

```bash
ls results/qwen3_8b_thinking_base_att_L*_adv200_judged/summary.json
```

Gate: choose the layer where base attack produces high coherent harmful action
without just turning into garbage. If multiple layers tie, prefer the one near
the smooth judged peak, not an isolated spike.

## 2. Train ABL-v8

```bash
cd /workspace/tamperforge
DL=<picked_layer> bash scripts/qwen3_8b_thinking_train_v8.sh
```

Default training is full-scope/all-layer AdamW, not 8-bit optimizer. It reduces
batch size and generative prompts to fit the 96GB card. If this OOMs, the first
fallback is:

```bash
DL=<picked_layer> TRAIN_SCOPE=last_half bash scripts/qwen3_8b_thinking_train_v8.sh
```

Do not switch to `adamw8bit` for this run unless we explicitly decide the
experiment is now "memory-fit probe" rather than full-precision scale test.

## 3. Snapshot pick

```bash
cd /workspace/tamperforge
DL=<picked_layer> STEM=outputs/tamper_resistant_qwen3_8b_thinking_v8.pt \
  bash scripts/qwen3_8b_thinking_pick_v8.sh
```

This runs `pick_v8_best.sh` with:

- `QWEN_THINKING=on`
- thinking-mode vLLM sampling: temperature 0.6, top-p 0.95, top-k 20
- `TF_IFEVAL_MAX_NEW=512` so the clean probe is not consumed by reasoning

The automatic pick is written to:

```text
results/auto_pick_v8_result.json
```

If it says `NO_SHIP`, retrain. Do not manually cherry-pick around a failed gate.

## 4. Four-cell eval

Set `BEST_CKPT` to the picked snapshot path from `auto_pick_v8_result.json`.

```bash
cd /workspace/tamperforge
DL=<picked_layer> BEST_CKPT=<picked_snapshot.pt> \
  bash scripts/qwen3_8b_thinking_eval_4cell.sh
```

This materializes and evaluates:

- `base_clean`
- `base_att`
- `v8_clean`
- `v8_att`

Safety is AdvBench-520 with DeepSeek judge. Capability is ARC, 12-subject MMLU,
IFEval, and GSM8K via lm-eval/vLLM.

## Notes

- vLLM safety generation is explicitly thinking-on. The generated text may
  contain `<think>...</think>` because this offline path does not use the API
  reasoning parser. Treat that as part of the model-visible output for this run.
- lm-eval IFEval uses `--apply_chat_template`; Qwen3 thinking is on by tokenizer
  default. The current harness path does not give us a clean per-command
  `enable_thinking=True` knob, so the runbook records this assumption.
- Snapshot storage is heavy. Full all-layer Qwen3-8B checkpoints can be around
  tens of GB each. Keep at least 1TB free or prune early failed snapshots.
