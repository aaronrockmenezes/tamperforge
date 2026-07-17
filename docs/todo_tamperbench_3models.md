# TODO: run TamperBench refusal_ablation on all 3 v8 models

Supersedes `docs/todo_tamperbench_blackwell.md` (gemma-v7-only, assumed a 96GB box was
required). Two things changed: (1) the product is now **ABL-v8**, not v7, for all three
architectures; (2) re-did the OOM math (below) and a 24GB card should work with one small
patch, not the 96GB Blackwell box that got killed after the Qwen3-8B run.

TamperBench [criticalml-uw, arXiv 2602.06911] = the standardized tamper-resistance bench (same
lab as AntiDote). Its `refusal_ablation` attack + StrongREJECT + MMLU-Pro is a **third-party,
fair** version of our abliteration result — running base vs v8 through it is the strongest
external validation of the ABL thread, across all 3 proven architectures.

## Why it OOM'd on 24GB before, and why the fix should now be enough

`refusal_ablation.py::_iso_get_last_position_logits` allocates
`out = torch.empty((N, V), dtype=torch.float64)` where N = full harmless-val set (~6264,
ignores the `data_samples` config knob) and V = the model's vocab size. Their loader also
reserves ~90% of visible VRAM. On a 24GB card that reservation caps usable memory around
21.6GB; the doc's original report was ~14GiB already in use (model + activations + loader
overhead) + ~12GiB for gemma's fp64 tensor > available budget.

**Fix: patch that one line's dtype `float64` -> `float32`, halving the tensor.** Recomputed
per-model (N≈6264 fixed, V = each model's actual vocab):

| model | vocab (V) | fp64 tensor (broken) | fp32 tensor (patched) |
|---|---|---|---|
| gemma-3-1b-it | 262144 | ~12.2 GiB | ~6.1 GiB |
| Qwen3-0.6B | 151936 | ~7.1 GiB | ~3.6 GiB |
| Llama-3.2-1B-Instruct | 128256 | ~6.0 GiB | ~3.0 GiB |

gemma is the worst case (biggest vocab): 6.1 GiB (patched) + ~14 GiB (baseline) ≈ 20.1 GiB,
comfortably under 24GB with ~4GB to spare. Qwen/Llama are both smaller models with smaller
vocabs, so their baseline "in use" figure will be lower than gemma's 14GiB too — more margin,
not less. **Target a 4090 (24GB), not a 96GB box.** If gemma specifically still comes up short
in practice, two backup levers are ready (see "if it's still tight" below) before reaching for
a bigger card.

## 0. Materialize all 3 v8 checkpoints as loadable HF dirs

Run once you have a box (any single 24GB+ GPU is fine for this step, it's just weight loading):

```bash
cd /workspace/tamperforge

python experiments/save_p1b_checkpoint.py --model-id google/gemma-3-1b-it \
  --checkpoint outputs/tamper_resistant_gemma3_1b_v8_best.pt --attack none --direction-layer 14 \
  --out /workspace/outputs/gemma_v8_hf

python experiments/save_p1b_checkpoint.py --model-id Qwen/Qwen3-0.6B \
  --checkpoint outputs/tamper_resistant_qwen3_0p6b_v8.pt --attack none --direction-layer 20 \
  --out /workspace/outputs/qwen_v8_hf

python experiments/save_p1b_checkpoint.py --model-id meta-llama/Llama-3.2-1B-Instruct \
  --checkpoint outputs/tamper_resistant_llama32_1b_v8_best.pt --attack none --direction-layer 13 \
  --out /workspace/outputs/llama_v8_hf

# add the deprecated key TamperBench's loader still requires, per model
for pair in "gemma_v8_hf:google/gemma-3-1b-it" "qwen_v8_hf:Qwen/Qwen3-0.6B" "llama_v8_hf:meta-llama/Llama-3.2-1B-Instruct"; do
  dir="${pair%%:*}"; base="${pair##*:}"
  python - "$dir" "$base" <<'EOF'
import json, sys
p = f"/workspace/outputs/{sys.argv[1]}/config.json"
c = json.load(open(p))
c["_name_or_path"] = sys.argv[2]
json.dump(c, open(p, "w"), indent=2)
EOF
done
```

(If the checkpoints aren't already on this box: pull from the private HF repo,
`aaronrockmenezes/tamperforge/adapters/`, or `outputs/` if this is the same box that trained
them.)

## 1. Install TamperBench + apply the 24GB patch

```bash
cd /workspace && git clone https://github.com/criticalml-uw/TamperBench.git
cd TamperBench && uv sync --all-groups

# the actual fix -- confirm the exact line number hasn't shifted upstream before patching
grep -n "dtype=torch.float64" refusal_ablation.py  # or wherever _iso_get_last_position_logits lives
sed -i 's/dtype=torch\.float64/dtype=torch.float32/' <path/to/that/file>
```

## 2. Patch grid.yaml: add model_config + MMLU-Pro eval

```bash
python - <<'EOF'
p = "configs/whitebox/attacks/refusal_ablation/grid.yaml"
s = open(p).read()
s = s.replace(
"""    max_generation_length: 512
    evals: [strong_reject, strong_reject_small]""",
"""    model_config:
        template: native
        max_generation_length: 512
        inference_batch_size: 16
    evals: [strong_reject, mmlu_pro_val]""")
open(p, "w").write(s)
print("patched" if "model_config" in s else "FAIL")
EOF
```

## 3. Run base vs v8, all 3 architectures

```bash
uv run scripts/whitebox/benchmark_grid.py google/gemma-3-1b-it              --attacks refusal_ablation --model-alias gemma_base
uv run scripts/whitebox/benchmark_grid.py /workspace/outputs/gemma_v8_hf    --attacks refusal_ablation --model-alias gemma_v8

uv run scripts/whitebox/benchmark_grid.py Qwen/Qwen3-0.6B                   --attacks refusal_ablation --model-alias qwen_base
uv run scripts/whitebox/benchmark_grid.py /workspace/outputs/qwen_v8_hf     --attacks refusal_ablation --model-alias qwen_v8

uv run scripts/whitebox/benchmark_grid.py meta-llama/Llama-3.2-1B-Instruct  --attacks refusal_ablation --model-alias llama_base
uv run scripts/whitebox/benchmark_grid.py /workspace/outputs/llama_v8_hf   --attacks refusal_ablation --model-alias llama_v8
```

## 4. (optional) worst-case adaptive sweep, per model

```bash
for m in gemma_v8 qwen_v8 llama_v8; do
  ckpt_dir="/workspace/outputs/${m%_v8}_v8_hf"
  uv run scripts/whitebox/optuna_single.py "$ckpt_dir" --attacks refusal_ablation --n-trials 50 --model-alias "$m"
done
```

## 5. Read results, all 3 side by side

```bash
python - <<'EOF'
import pandas as pd, glob
for m in ["gemma_base","gemma_v8","qwen_base","qwen_v8","llama_base","llama_v8"]:
    print("\n===", m, "===")
    for f in sorted(glob.glob(f"results/grid_*/{m}/**/results.parquet", recursive=True)):
        df = pd.read_parquet(f); tag = f.split(m+"/")[1].split("/")[0]
        print(tag, dict(zip(df.metric_name, df.metric_value.round(4))))
EOF
```

## If it's still tight on 24GB (gemma specifically)

Two more levers before reaching for a bigger box, both already identified in the original
doc: (a) lower the loader's `max_memory`/reservation fraction so more of the 24GB is actually
usable instead of soft-capped at 90%, or (b) trim `harmless_val` sample count in their config
(the `data_samples` knob is apparently ignored by their code -- may need a direct data-loading
patch, not just a config change). Try (a) first, it's a one-line change; only fall back to a
bigger box if both (a) and (b) together still aren't enough.

## Expected result

base_att -> high StrongREJECT harm + MMLU-Pro intact; v8_att -> ~0 harm + MMLU-Pro crater
= MAD confirmed in a third-party harness, across all 3 architectures, on the current product
(not the superseded v7).
