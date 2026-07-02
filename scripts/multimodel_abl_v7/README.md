# Multimodel + multiseed ABL-v7 — 4×3090 orchestration

ABL-v7 = attack-ensemble version of `experiments/train_tamper_resistant.py`: full-scope
training, all-layer abliteration, argmax gibberish objective, random attack scope/layer
subsets per step. **Canonical recipe (from `results/p1b_a_ensemble_v7/manifest.json`):**
`--train-scope all --attack-ensemble --gib-mode argmax --lambda-gib 4 --lambda-uncensor 4
--lambda-safe 1 --lambda-reg 0.1 --steps 500 --lr 1e-5`. All scripts here use it EXACTLY
(direction-layer is per-model). Do not drift — multi-model + multi-seed must share config.

## 4-card layout (24GB each; all-scope 1.7B fits ~20–22GB)
| card | job | script |
|---|---|---|
| 0 | Qwen3-1.7B ABL-v7 | `run_qwen3_1p7b_v7.sh` (direction-layer 14) |
| 1 | Llama-3.2-1B ABL-v7 | `run_llama32_1b_v7.sh` (direction-layer 8) |
| 2 | gemma-3-1b ABL-v7 **seed 1** | `SEED=1 CUDA_VISIBLE_DEVICES=2 run_gemma3_1b_v7_seed.sh` |
| 3 | gemma-3-1b ABL-v7 **seed 2** | `SEED=2 CUDA_VISIBLE_DEVICES=3 run_gemma3_1b_v7_seed.sh` |

cards 0/1 = generality (multi-model). cards 2/3 = rigor (multi-seed; seed 42 = existing
product, so this makes n=3). Four birds, one box.

## 0. Setup (once, on the fresh box)
```bash
cd /workspace/tamperforge && source /venv/main/bin/activate   # or the box's env
git pull origin main
# gemma + Llama are GATED — must have approved HF access:
python -c "from huggingface_hub import auth_check; \
  [auth_check(m) for m in ('google/gemma-3-1b-it','meta-llama/Llama-3.2-1B-Instruct','Qwen/Qwen3-1.7B')]; \
  print('ALL MODELS ACCESSIBLE')"
```
If the Llama line 401s → request access on its HF page / fix `HF_TOKEN`, else card1 dies at load.

## 1. Smoke each FIRST (confirm 24GB fits all-scope 1.7B, ~4 steps)
```bash
# Qwen smoke (card0)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 \
python experiments/train_tamper_resistant.py --model-id Qwen/Qwen3-1.7B \
  --out outputs/_smoke_qwen.pt --run-id _smoke_qwen --train-scope all --abliterate-layers all \
  --attack-ensemble --direction-layer 14 --gib-mode argmax --gib-gen-tokens 32 --gib-gen-prompts 2 \
  --lambda-gib 4 --lambda-uncensor 4 --lambda-safe 1 --lambda-reg 0.1 --steps 4 --eval-every 2 --lr 1e-5 --seed 42
# Llama smoke (card1): same but --model-id meta-llama/Llama-3.2-1B-Instruct --direction-layer 8 \
#   --run-id _smoke_llama --out outputs/_smoke_llama.pt   and CUDA_VISIBLE_DEVICES=1
# gemma smoke (card2): same but --model-id google/gemma-3-1b-it --direction-layer 13 \
#   --run-id _smoke_gemma --out outputs/_smoke_gemma.pt   and CUDA_VISIBLE_DEVICES=2
```
**Smoke green:** no OOM; `L_safe` low (clean refusal intact); `ref_abl` high/rising;
`gib_ce` moving toward ~4. Qwen1.7B all-scope peaks ~20–22GB — if it OOMs, this box is
undersized: fall back `--train-scope last_half` (document the deviation) or ask for 8-bit AdamW.

## 2. Full runs (after all smokes green) — 4 parallel
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/multimodel_abl_v7/run_qwen3_1p7b_v7.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/multimodel_abl_v7/run_llama32_1b_v7.sh
SEED=1 CUDA_VISIBLE_DEVICES=2 bash scripts/multimodel_abl_v7/run_gemma3_1b_v7_seed.sh
SEED=2 CUDA_VISIBLE_DEVICES=3 bash scripts/multimodel_abl_v7/run_gemma3_1b_v7_seed.sh
```
One script per tmux pane (user watches the box; do NOT background/nohup). ~500 steps each.
Sanity per step-25/50 above; `[ablated gen]` should start collapsing to gibberish by
~step 75–150 = the entanglement forming.

## 3. After training — validate (the actual result, NOT in these scripts)
Each ckpt needs the abliteration battery to prove the claim: materialize → abliterate →
judge (ASR + gibberish) + lm_eval ARC/MMLU, clean vs attacked, vs base. Same protocol as
gemma ABL-v7. Win = abliterating the trained model → ~0 ASR/gibberish + capability drop,
while abliterating the base model → coherent harm. That reproduces the gemma result on 3
models + gives seed variance. Ask for the per-model battery commands when the ckpts land.

## Outputs
`outputs/tamper_resistant_{qwen3_1p7b,llama32_1b}_v7.pt`,
`outputs/tamper_resistant_p1b_v7_seed{1,2}.pt`. Back up to HF after (`push_to_hf.py`).
