# Run scripts

Use `train_version_g_pro6000.sh` for training-only fleet runs and
`eval_version_g_adaptive.sh` for post-training fresh-layer attacks and standard evaluation.
`version_g_final.sh` remains the historical all-in-one campaign entrypoint.

The final estimator defaults to `arditi_residual`. The adaptive evaluator also runs one external
Obliteratus `aggressive` arm from a pinned official Git checkout; it intentionally does not use
the placeholder PyPI package or reinterpret Obliteratus “surgical” as TamperForge `capK=16`.

`chain_g_new_gates.sh` is a compatibility alias. `chain_f.sh`, `chain_2gpu.sh`, and the
model/version-specific launchers are historical reproduction scripts; do not compose new
campaigns from them.

The canonical launcher and rank terminology are documented in
`docs/version_g_final.md`.
