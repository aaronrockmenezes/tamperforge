# P1b-A attack battery — validating the defense

P1b-A (v6) PASSED one attack (empirical refusal dir, scope=all matching training):
judge-ASR 0.844 (base) → 0.032 (trained). To be a real defense it must survive a
battery of DIVERSE attacks. Checkpoint: `outputs/tamper_resistant_p1b_v6.pt`
(git + HF). We will do all of these.

Read gate for every attack: **trained_attacked judge-ASR stays LOW** (near the
0.032 matched-attack level). If any attack pushes ASR back toward base's 0.844,
the entanglement is specific to what that attack happened to match, and the
defense needs to cover a broader subspace (train against an ensemble).

Fast path: the P1b-A model is pure weights → `save_p1b_checkpoint.py` materializes
each attacked variant as an HF dir → `p0_baseline_eval.py --backend vllm` (batched)
→ judge locally. No slow unbatched HF generation.

## Tier 1 — abliteration variants (code READY)

`experiments/save_p1b_checkpoint.py` supports all of these:

1.1 **Scope-mismatch** — attacker ablates a different scope than we trained (all).
   `--attack mlp` (we trained all), or `--direction-layer 9` (d from another layer).
   Status: run in progress.

1.2 **Layer-subset** — attacker doesn't hit all 26 layers.
   `--attack all --abliterate-layers 13-25` (or `0-12`).

1.3 **Rank-k subspace** — attacker ablates a refusal SUBSPACE, not one direction.
   `--attack all --attack-estimator svd --attack-rank 4` (or 8).

1.4 **Different direction estimator** — SVD / whitened-SVD instead of diff-in-means,
   and a different prompt sample for d.
   `--attack-estimator svd|whitened_svd` (+ `--attack-rank k`), `--direction-seed 7`.

Pipeline per variant:
```bash
python experiments/save_p1b_checkpoint.py --checkpoint outputs/tamper_resistant_p1b_v6.pt \
  --attack all --attack-estimator whitened_svd --attack-rank 4 --out outputs/p1b_v6_att_wsvd4
python experiments/p0_baseline_eval.py --backend vllm --model-id outputs/p1b_v6_att_wsvd4 \
  --advbench-source walledai --n-advbench 200 --n-arc 0 --max-new-tokens 128 \
  --max-length 4096 --run-id p1b_v6_att_wsvd4_gen
# then judge_generations.py locally
```

Always also save the **base** control with the same attack (`--attack ... ` no
`--checkpoint`) so the comparison is matched.

## Tier 2 — different attack CLASS (code TBD)

2.1 **Fine-tuning attack (P4) — highest priority.** Load the trained model, FT on N
   harmful examples (few-shot), measure if harmful capability returns WITHOUT the
   gibberish. Abliteration-resistance ≠ FT-resistance. Compare vs TAR / RepNoise.
   Needs a new script `experiments/ft_attack.py`.

2.2 **Steering-vector attack.** Add a refusal-ablating steering vector at inference
   (Turner 2023 / Rimsky 2024) instead of editing weights. Our defense targets
   weight ablation — does it hold vs activation steering? Needs an inference-time
   steering hook + sweep over alpha.

2.3 **OBLITERATUS (P2).** Adaptive attacker: whitened-SVD, per-layer, cluster-aware
   layer selection, entanglement-gated skipping, Ouroboros self-repair. Built to
   dodge this defense. SEPARATE harness (AGPL-3.0 — pip-install + run on the saved
   checkpoint; never vendor its code).

## Rigor (parallel, for a writeup)

- **Seeds**: retrain v6 with 2–3 seeds — is it a lucky run?
- **Capability beyond ARC**: MMLU + GSM8K via lm_eval on the saved dirs.
- **Generality**: Qwen3-1.7B, Llama-3.2-1B.
- **Ablations**: which of {argmax gib, attention scope, λ_uncensor} is load-bearing?
  (prose-gib + mlp-only failed; isolate the necessary pieces.)

## Order

1. Finish Tier 1 (1.1 running → 1.2, 1.3, 1.4). Fast, decides if the defense is
   direction/scope-specific.
2. Build + run the **fine-tuning attack (2.1)** — the decisive realistic test.
3. Seeds + MMLU (rigor).
4. Steering (2.2) + OBLITERATUS (2.3) — adaptive-attacker capstone.
5. If any Tier-1/2 attack leaks → **harden**: train against an ensemble of attacks
   (multiple d's / scopes / estimators), and adversarial FT-resistance (TAR-style).
