# Multimodel ABL-v7 runs

ABL-v7 means the attack-ensemble version of `experiments/train_tamper_resistant.py`:
full-scope training, all-layer abliteration, argmax gibberish objective, and random
attack scope/layer subsets during training.

Foreground launch scripts:

```bash
bash scripts/multimodel_abl_v7/run_qwen3_1p7b_v7.sh
bash scripts/multimodel_abl_v7/run_llama32_1b_v7.sh
```

Defaults:

- Qwen: `CUDA_VISIBLE_DEVICES=0`, `Qwen/Qwen3-1.7B`, direction layer 14.
- Llama: `CUDA_VISIBLE_DEVICES=1`, `meta-llama/Llama-3.2-1B-Instruct`, direction layer 8.
- Outputs go to `outputs/tamper_resistant_{qwen3_1p7b,llama32_1b}_v7.pt`.

Step-25/50 sanity:

- `L_safe` should stay low enough that clean refusal is not lost.
- `ref_abl` should rise or remain high.
- `gib_ce` should move toward or above 4.
- By roughly step 75-150, `[ablated gen]` should start collapsing.
