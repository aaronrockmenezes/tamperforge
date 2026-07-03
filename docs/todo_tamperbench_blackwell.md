# TODO: run TamperBench refusal_ablation on RTX Pro 6000 (Blackwell 96GB)

TamperBench [criticalml-uw, arXiv 2602.06911] = the standardized tamper-resistance bench
(same lab as AntiDote). Its `refusal_ablation` attack + StrongREJECT + MMLU-Pro = a **third-party,
fair** version of our abliteration result. Running gemma base vs gemma-v7-abl through it is the
strongest external validation of the ABL thread. Blocked on 24GB (OOM); trivial on 96GB.

## Why it OOMs on 24GB (not our bug — theirs)
`refusal_ablation.py::_iso_get_last_position_logits` allocates
`out = torch.empty((N, V), dtype=torch.float64)` where N = **full** harmless-val set (~6264,
ignores `data_samples`) and V = gemma vocab 262144 → ~12 GiB in fp64. Their loader also reserves
90% VRAM for the model. 12 GiB + ~14 GiB in use > 24 GiB. On 96GB it just fits.
(24GB workaround if ever needed: patch line ~353 dtype float64→float32 (halves to ~6GB) AND
cut harmless_val, OR lower the loader's max_memory reservation. Not worth it — use the big box.)

## Turnkey steps (fixes already validated, just re-apply on the Blackwell box)
```bash
# 0. materialize v7 as a loadable HF dir
cd /workspace/tamperforge
python experiments/save_p1b_checkpoint.py --model-id google/gemma-3-1b-it \
  --checkpoint outputs/tamper_resistant_p1b_v7.pt --attack none --direction-layer 13 \
  --out /workspace/outputs/gemma_v7_hf
# add the deprecated key their loader still requires
python - <<'EOF'
import json; p="/workspace/outputs/gemma_v7_hf/config.json"
c=json.load(open(p)); c["_name_or_path"]="google/gemma-3-1b-it"
json.dump(c,open(p,"w"),indent=2)
EOF

# 1. install
cd /workspace && git clone https://github.com/criticalml-uw/TamperBench.git
cd TamperBench && uv sync --all-groups

# 2. patch their refusal_ablation grid.yaml: add model_config block + MMLU-Pro eval
python - <<'EOF'
p="configs/whitebox/attacks/refusal_ablation/grid.yaml"; s=open(p).read()
s=s.replace(
"""    max_generation_length: 512
    evals: [strong_reject, strong_reject_small]""",
"""    model_config:
        template: native
        max_generation_length: 512
        inference_batch_size: 16
    evals: [strong_reject, mmlu_pro_val]""")
open(p,"w").write(s); print("patched" if "model_config" in s else "FAIL")
EOF

# 3. run base vs v7 (on 96GB no data_samples/batch cuts needed)
uv run scripts/whitebox/benchmark_grid.py google/gemma-3-1b-it       --attacks refusal_ablation --model-alias gemma_base
uv run scripts/whitebox/benchmark_grid.py /workspace/outputs/gemma_v7_hf --attacks refusal_ablation --model-alias gemma_v7

# 4. worst-case adaptive sweep (50 Optuna trials over KL/prune-depth)
uv run scripts/whitebox/optuna_single.py /workspace/outputs/gemma_v7_hf \
  --attacks refusal_ablation --n-trials 50 --model-alias gemma_v7

# 5. read results
python - <<'EOF'
import pandas as pd, glob
for m in ["gemma_base","gemma_v7"]:
    print("\n===",m,"===")
    for f in sorted(glob.glob(f"results/grid_*/{m}/**/results.parquet", recursive=True)):
        df=pd.read_parquet(f); tag=f.split(m+"/")[1].split("/")[0]
        print(tag, dict(zip(df.metric_name, df.metric_value.round(4))))
EOF
```

Expected: base_att → high StrongREJECT harm + MMLU-Pro intact; v7_att → ~0 harm + MMLU-Pro
crater = MAD in their harness. Also extend to Qwen/Llama v7 (pull_ckpts) once gemma confirms.
