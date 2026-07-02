# Multimodel ABL-v1 runs

Two foreground launch scripts for the 2x5090 box:

```bash
bash scripts/multimodel_abl_v1/run_qwen3_1p7b.sh
bash scripts/multimodel_abl_v1/run_smollm2_1p7b.sh
```

Defaults:

- Qwen uses `CUDA_VISIBLE_DEVICES=0`, `Qwen/Qwen3-1.7B`, direction layer 14.
- SmolLM2 uses `CUDA_VISIBLE_DEVICES=1`, `HuggingFaceTB/SmolLM2-1.7B-Instruct`, direction layer 12.
- Qwen thinking mode is disabled in `tamperforge.model.apply_chat_template_no_think`
  via `enable_thinking=False` when supported by the tokenizer.

Step-25/50 sanity:

- `L_safe` should be low or falling.
- `ref_abl` should be high.
- `gib_ce` should move toward/above 4.
- By roughly step 75-150, `[ablated gen]` should start collapsing.
