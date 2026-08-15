# Obliteratus attacker review — 2026-08-15

Reviewed official source commit `885390a0e2d78dfa9a62edaea5739d28d2b6903d` and the published
`obliteratus==0.0.1` wheel. The wheel contains only a name-reservation placeholder; the GitHub
tree reports version `0.1.2` and is the implementation reviewed here. TamperForge therefore runs
the upstream source at an exact commit without vendoring it.

## Decision

Use exactly one package-native external arm in the standard adaptive chain: `aggressive`.
It is the best useful contrast to TamperForge's fixed rank attacks because it estimates separate
rank-8 subspaces by layer, whitens against harmless covariance, selects/weights layers
adaptively, re-probes after editing, adds jailbreak contrast, winsorizes activations, and edits
selected attention heads. Do not treat it as another spelling of our rank-8 attack.

## Method inventory

| method | core edit | use here? |
|---|---|---|
| `basic` | one difference-of-means direction; plain Arditi-style projection | redundant control |
| `advanced` | rank-4 SVD, norm preservation, 30% regularization, adaptive strengths | useful later sensitivity |
| `aggressive` | rank-8 whitened SVD plus adaptive/iterative/jailbreak/head features | **default external arm** |
| `spectral_cascade` | rank-6 whitened SVD plus DCT bands over layer depth | exploratory only |
| `informed` | advertised analysis-guided pipeline | blocked: CLI instantiates the ordinary pipeline |
| `surgical` | MoE/neuron/head/SAE targeting | distinct from TamperForge surgical cap-rank |
| `optimized` | nominal Optuna search over attack strengths | only if `optuna` is installed; otherwise upstream skips it |
| `inverted` | reflection rather than projection | too destructive for the primary comparable threat |
| `nuclear` | reflection plus embeddings, experts, and steering | stress test, not comparable abliteration |
| `rdo` | rank-4 SVD then gradient-refined refusal directions | interesting secondary arm; absent from upstream CLI choices |
| `som` | self-organizing-map direction manifold | broken at this commit: referenced module is absent |

## Important semantic differences

- Obliteratus SVD pairs harmful and harmless examples and decomposes their activation
  differences. TamperForge's current `svd` centers harmful activations by a shared benign mean.
- Obliteratus `aggressive` learns a different subspace at each selected layer. TamperForge's
  fixed rank-k attacks learn one basis at the chosen read layer and apply it across decoder
  projections.
- Obliteratus `surgical` means sparse MoE/neuron/head/SAE targeting. TamperForge surgical means
  removing the top capability-PC components from the refusal basis before applying it.
- Upstream has broad architecture fallback support, but its explicit projection names do not
  cover every TamperForge Qwen3.5 linear-attention matrix. Each new architecture still needs a
  real attacked-model materialization smoke before campaign-scale evaluation.

## Reproducibility and safety controls

- Pin repo URL and full Git SHA in the evaluator and record them beside the output model.
- Set `OBLITERATUS_TELEMETRY=0`; do not pass `--contribute`.
- Fail on the known-broken `som` and misleading `informed` CLI paths.
- Fail `optimized` when Optuna is absent instead of accepting silent fallback.
- Keep attacked weights temporary. Preserve generations, judged summaries, logs, and provenance.
- Do not replace TamperForge rank/surgical/Heretic arms with this external arm; it is additive.
