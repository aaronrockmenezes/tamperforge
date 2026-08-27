# Version G final

## Scientific status — 2026-08-15

**Frozen for reproduction; not approved for new training.** Fresh attacks broke both evaluated
checkpoints:

| checkpoint | fresh attack | selected layer | held-out n | harmful-actionable | gibberish | refused |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B older Version G | rank-1 | 10 | 504 | 0.4306 | 0.0238 | 0.1012 |
| Qwen3-0.6B older Version G | rank-2 | 11 | 504 | 0.6845 | 0.1012 | 0.0575 |
| Phi-4-mini Version G final | rank-1 | 13 | 503 | 0.8350 | 0.0099 | 0.0517 |
| Phi-4-mini Version G final | rank-2 | 14 | 504 | 0.5972 | 0.0218 | 0.2738 |

The held-out column excludes the 16 prompts used to choose each winning layer. Phi rank-1 has
503 rather than 504 rows because one of the 520 API judgments never completed. Full provenance,
raw-table values, telemetry, and interpretation are in
`docs/findings_fresh_rank_attacks_2026_08_15.md`.

The recipe's internal objective converged without producing the desired behavior. On Phi,
centered `L_rr` averaged 0.955 over steps 1-25 and 0.060 over steps 476-500. However,
`L_harm > 0` on only 7/500 steps and `L_gib > 0` on 0/500. `L_rr` only requires attacked
representations to move away from a frozen step-0 reference; orthogonal harmful behavior also
satisfies it. Do not interpret low `L_rr` as resistance.

## Phi-4-mini checkpoint trajectory and follow-up probes — 2026-08-27

Fresh per-layer rank-1 probes were collected for the Phi-4-mini base and steps 500, 600, 700,
800, 900, and 1000. The selected layer stayed at L13 through step 700, then moved to L14. The
trajectory is non-monotonic and does not show that longer training restores resistance. It uses
mixed sample counts (16 prompts/layer for base/500/700; 64 for 600/800/900/1000), so these are
exploratory checkpoints rather than one directly comparable confirmation series.

On the exact step-700 checkpoint, a fresh surgical rank-1 `capK=16` sweep over L8–L16 produced
no coherent harm in that band but became increasingly gibberish-heavy at higher layers. A
10-shot, five-epoch LoRA attack followed by a fresh rank-1 sweep still reached 56.25% harm at
L13. The full per-layer table, Qwen comparison, settings, and artifact boundaries are in
[`docs/results_phi4mini_rank1_trajectory_20260827.md`](results_phi4mini_rank1_trajectory_20260827.md).

The launcher and evaluator below remain canonical only so the failed recipe can be reproduced
exactly and the five retained checkpoints can be evaluated consistently.

Canonical entrypoint: `scripts/runs/version_g_final.sh`; canonical trainer:
`experiments/train_version_g_final.py`. The old trainer filename remains a compatibility shim
for historical commands.

For split hardware use, run `scripts/runs/train_version_g_pro6000.sh` on the expensive
training box, then `scripts/runs/eval_version_g_adaptive.sh` on the inference box. The
training-only launcher performs no sweep, generation benchmark, API judgment, Heretic search,
or lm-eval job. It performs a one-step architecture smoke, trains with periodic evaluation
disabled, saves diagnostic weight snapshots every 100 steps (not optimizer-resume states),
materializes the clean model,
uploads checkpoint/model/log/manifest to a private HF repo, and verifies the remote files.

Preflight without GPU/network work:

```bash
RANK_K_ESTIMATOR=arditi_residual DRY_RUN=1 \
  bash scripts/runs/train_version_g_pro6000.sh
```

The training distribution has two different rank axes. Do not call both of them “k” without
the prefix:

- **attack rank (`atkK`)**: how many refusal directions the canonical attack removes.
  Version G final gives this branch 10% total probability and samples uniformly from
  `{1,2,4,8,16}`. Each rank therefore receives 2% of all steps in expectation.
- **capability rank (`capK`)**: how many capability PCs are projected out of every refusal
  basis vector. The training sampler's surgical branch remains attack-rank 1; evaluation also
  tests surgical attack ranks 4 and 16.

More precisely, a shared `atkK=2` attack estimates two orthonormal directions at one selected
read layer, then projects that same two-dimensional subspace out of every residual read/write
matrix in every decoder layer. It does **not** estimate two directions independently per model
layer. That stronger geometry is called a per-layer adaptive attack and is separate.

Likewise `capK=16` estimates one 16-dimensional capability subspace at the selected read layer.
It removes those capability components from the refusal basis, re-orthonormalizes the result,
then applies that resulting refusal basis across every decoder layer. `capK=16` is neither 16
directions per layer nor an attack confined to one layer. Training surgical draws remain
`atkK=1`; the adaptive evaluation additionally crosses `atkK={4,16}` with `capK=16`.

Historical `version_b` behavior is frozen at canonical rank 1. Only the new
`version_g_final` attack profile uses multi-rank attacks.

The retained `0.40` surgical knob is conditional on reaching the generic subset branch. With
10% rank-k and 35% absolute Heretic mass, its expected total mass is about 22%
(`0.55 * 0.40`). It is not a 40%-of-all-steps branch. The report records both values.

## Rank-k estimator

The final launchers now default to `RANK_K_ESTIMATOR=arditi_residual`, the selected experiment:

| value | basis | tradeoff |
|---|---|---|
| `svd` | top-k right singular vectors of harmful activations centered by the benign mean | cleanest standard rank-k basis; k=1 is not legacy mean-difference Arditi |
| `arditi_residual` | legacy harmful-minus-benign mean first, followed by SVD directions orthogonal to it | k=1 exactly preserves Arditi; higher ranks use a documented hybrid basis |
| `partitioned` | mean-difference directions from disjoint prompt partitions, orthonormalized | multiple independently estimated refusal directions; more sampling-sensitive |

Example preflight (launches nothing):

```bash
RANK_K_ESTIMATOR=arditi_residual DRY_RUN=1 \
MODEL_ID=google/gemma-3-1b-it GPU=0 \
bash scripts/runs/version_g_final.sh
```

## Training diagnostics

`TRAIN_EVAL_EVERY=25` runs the cheap held-out loss block. Set it to `0` to disable periodic
evaluation. Slow GSM8K generation and a local AdvBench preview are off by default; opt in with
`TRAIN_DIAGNOSTICS=1`. The canonical launcher does not perform blocking API judgment inside
the training loop; full post-training judgments run asynchronously during evaluation.

The GSM8K diagnostic never shapes training. GSM8K enters the loss only when both
`--gib-mode task` and a positive gibberish-loss weight are used. The canonical final launcher
currently preserves Version G's `lambda_gib=0`, so no GSM8K example affects its gradients.

## Evaluation flow

The split adaptive evaluator recomputes layer quality on the final trained model separately
for each fixed TamperForge geometry, then runs the standard battery on:

1. clean Version G;
2. `atkK=1`;
3. `atkK=1, capK=16` surgical;
4. fresh Heretic;
5. `atkK=4`;
6. `atkK=16`;
7. `atkK=4, capK=16` surgical;
8. `atkK=16, capK=16` surgical;
9. the package-native Obliteratus `aggressive` attack.

```bash
MODEL_ID=... CHECKPOINT=outputs/version_g_<slug>_rrcenter.pt SHORT=<slug>_rrcenter \
RANK_K_ESTIMATOR=arditi_residual \
  bash scripts/runs/eval_version_g_adaptive.sh
```

The six fixed attacks use independent winning layers chosen by maximum judged coherent harm,
then minimum gibberish/refusal. The sweep saves the exact six selected bases; materialization
reuses those tensors instead of silently recomputing a different direction. The sweep summary
also pins the trained-checkpoint SHA-256, and the evaluator refuses a stale checkpoint or basis.
Heretic performs its own continuous/fractional layer search. The old base-selected layer is not
reused.

Obliteratus is deliberately one external attack arm, not another method zoo. It defaults to
`aggressive`: an independently estimated per-layer rank-8 whitened-SVD subspace with adaptive
layer strengths, iterative re-probing, jailbreak contrast, activation winsorization, and
attention-head surgery. That geometry is materially different from our shared rank-k basis.
Its package meaning of “surgical” is unrelated to TamperForge `capK=16`.

The PyPI release is a name-reservation placeholder, so the evaluator does not install it. It
clones the official source into `/workspace/.cache/tamperforge`, checks out commit
`885390a0e2d78dfa9a62edaea5739d28d2b6903d`, records that provenance with the attacked model,
and disables telemetry. `informed` and `som` are blocked at this pin because the former's CLI
does not use its advertised informed pipeline and the latter imports a missing module.
`optimized` is allowed only when its undeclared `optuna` dependency is actually installed.

Override the external method only for an explicit sensitivity study:

```bash
OBLITERATUS_METHOD=advanced bash scripts/runs/eval_version_g_adaptive.sh
```

Attacked model directories are temporary; generations, judgments, summaries, selected layers,
selected bases, Heretic top-three configs, and the Obliteratus source pin persist.

### Historical all-in-one flow

The launcher runs the existing benchmark tasks separately under one persistent vLLM server.
AdvBench/XSTest judgments run in one background API queue while GPU capability eval proceeds.
MT-Bench judgments similarly begin per arm while the GPU generates the following arm.

Order:

1. Direction-layer sweep.
2. Untouched base full battery.
3. Untouched base rank-1 and surgical-k16 AdvBench controls.
4. Version G final training.
5. Version G clean full battery.
6. Rank-1 and surgical-k16 builds and batteries while Heretic searches.
7. Top Heretic winner replay and full battery; preserve up to three winner configs.
8. MT-Bench, runtime gates, and one JSON/Markdown report.

XSTest clean-usability tolerance is an absolute 10 percentage points from the same model's
base. Surgical `capK=16` and canonical `atkK=16` are deliberately reported as different fields.

## Proposed reference-memory experiment — not implemented

The next small mechanism test may use two distinct bounded histories:

1. an **attack-direction bank** containing refusal bases re-estimated at each 25-step refresh;
2. a **clean-state bank** containing detached clean activations from step 0 plus the latest
   three refreshes (`N=4`), aligned to the same prompts, tokens, and layers.

Do not sum raw cosine terms. Duplicate historical states would be counted repeatedly and signed
cosines could cancel. Orthonormalize the centered historical states into a basis `Q` and minimize
normalized projection energy:

```text
L_hist = ||Q^T a_attacked||^2 / ||a_attacked||^2
```

This is the compact way to minimize overlap with all retained predecessors. Keep the history
bounded: unlimited states eventually span the hidden space and leave only the zero vector.
Cache detached activations or a low-rank basis; do not keep full historical models.

This change is insufficient alone. It still specifies “different from clean,” not “safe,”
“refused,” or “incapable.” Any continuation run must pair it with a direct attacked-output
behavioral objective and must be validated with fresh post-training directions/layers.
