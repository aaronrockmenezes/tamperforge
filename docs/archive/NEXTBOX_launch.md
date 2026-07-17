# Next-box launch cheatsheet (8x3090) — turnkey

Setup: clone repo, `/venv/main` or install deps, put OPENROUTER_API_KEY in `.env`, HF login
(gemma/llama gated). Pull v8 ckpts from HF adapters/ as needed. Judge LOCAL via env_ml.
8 GPUs → run per-GPU jobs in parallel (CUDA_VISIBLE_DEVICES=0..7).

## Per-model direction-layers (judged sweep peaks — DON'T guess)
gemma **14** (peak 0.890; L13-15 plateau, v7 used 13) · Qwen 20 (v7 used 14) · Llama 13 ·
SmolLM2 ~15 (sweep peak, verify tail L16-23).

## 1. gemma v8 (the 3/3 tiebreaker) — GPU 0
```bash
MODEL=google/gemma-3-1b-it DL=14 OUT=outputs/tamper_resistant_gemma3_1b_v8.pt \
  CUDA_VISIBLE_DEVICES=0 bash scripts/train_v8.sh
# then: prune stage-1 snapshots, pick:
rm outputs/tamper_resistant_gemma3_1b_v8.pt.s{25,50,75,100,125,150,175,200,225}.pt
MID=google/gemma-3-1b-it STEM=outputs/tamper_resistant_gemma3_1b_v8.pt DL=14 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/pick_v8_best.sh
# judge results/pk_*_{att,clean}_adv200 locally -> 4-axis MANUAL pick -> promote to *_v8_best.pt
# AND (parallel, automated, pre-registered gates) — validates/reproduces the manual pick:
#   STEM=outputs/tamper_resistant_gemma3_1b_v8.pt bash scripts/auto_pick_v8.sh   (run LOCAL)
#   gates: att_harm<=0.05, att_gib>=0.90, clean_harm<=0.10 -> max clean_cap. Selection on
#   AdvBench-200 (val); report FINAL on full test suite. No survivor => exits 2 (NO SHIP) +
#   per-snapshot failure report. Reproduces gemma s450 / llama s425. Result JSON persisted.
```
If clean leaks harm (like llama first run) it's already handled by S2SAFE=4 default.

## 2. Full 9-bench matrix per model (harm+cap) — reuse eval_matrix_{qwen,llama}.sh
Qwen + Llama done (`full_eval_matrices/`). For gemma: copy eval_matrix_llama.sh -> gemma
(MID=google/gemma-3-1b-it, V7=outputs/tamper_resistant_p1b_v7.pt, V8=gemma v8_best, DL=13).

## 3. Extended suite (over-refusal/SimpleQA/MBPP) on llama + gemma — parallel GPUs
Fetch prompts once: `python scripts/external_benches/fetch_prompts.py` (xstest/orbench/simpleqa).
```bash
# llama (GPU 1)
MID=meta-llama/Llama-3.2-1B-Instruct TAG=ll V7=outputs/tamper_resistant_llama32_1b_v7_L13.pt \
  V8=outputs/tamper_resistant_llama32_1b_v8_best.pt DLBASE=13 DLV7=13 DLV8=13 \
  CUDA_VISIBLE_DEVICES=1 bash scripts/eval_matrix_new.sh
# gemma (GPU 2) — after gemma v8 exists. DL: base/v8 attack @14 (peak), v7 @13 (its trained layer)
MID=google/gemma-3-1b-it TAG=gm V7=outputs/tamper_resistant_p1b_v7.pt \
  V8=outputs/tamper_resistant_gemma3_1b_v8_best.pt DLBASE=14 DLV7=13 DLV8=14 \
  CUDA_VISIBLE_DEVICES=2 bash scripts/eval_matrix_new.sh
```
Score locally: score_overrefusal.py (xstest/orbench), score_simpleqa.py, mbpp self-scored.

## 4. Scale-up (the AntiDote gap) — bigger models, adamw8bit or pooled mem
SmolLM2-1.7B (fits 24GB w/ adamw8bit), Phi-4-mini-3.8B, Ministral-3B. Sweep DL first
(save_p1b_checkpoint --attack all --direction-layer $L, p0 advbench 200, judge), then train_v8.sh.

## 5. Seed robustness — 8 GPUs = 8 seeds trivially
Same train_v8.sh with SEED=1..8, pick best per seed, report win-rate.

## Gotchas (learned 2026-07-04)
- MBPP needs HF_ALLOW_CODE_EVAL=1 (in eval_matrix_new.sh) + sandbox (vast box ok).
- score_overrefusal/simpleqa load .env + assert OPENROUTER_API_KEY (429 silently fakes labels).
- --save-every writes ~1.9G/ckpt; prune stage-1 (.s25..s225) before pick.
- Safety judge MISLABELS gibberish as benign on benign-framed prompts -> use score_overrefusal.py.
- v8 training OSCILLATES in stage 2 -> the save-every 4-axis pick is REQUIRED (final step is a lottery).
- Per-model DL by judged sweep, not 50%-depth guess.
See docs/devlog_2026_07_04.md + handoff_2026_07_03_MASTER.md.
