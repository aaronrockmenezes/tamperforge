# Gemma-3-1b v8 snapshot pick (2026-07-04) — 4-axis, DL=14 (sweep peak)
Two-stage v8, lambda_gib 8, lambda_clean 3, stage2-lambda-safe 4, stage2-lambda-gib 4, save-every 25.
AdvBench 200 judged (clean & attacked); clean_cap = 24-prompt instruction probe.

| snap | clean_harm | clean_cap | att_harm | att_gib |
|---|--|--|--|--|
| s250 | 0.000 | 0.208 | 0.000 | 1.00 |
| s275 | 0.000 | 0.208 | 0.000 | 1.00 |
| s300 | 0.000 | 0.458 | 0.000 | 1.00 |
| s325 | 0.120 | 0.542 | 0.730 | 0.01 |
| s350 | 0.055 | 0.542 | 0.830 | 0.01 |
| s375 | 0.070 | 0.542 | 0.815 | 0.00 |
| s400 | 0.050 | 0.583 | 0.900 | 0.01 |
| s425 | 0.145 | 0.667 | 0.000 | 1.00 |
| s450 | 0.080 | 0.625 | 0.000 | 1.00 |
| s475 | 0.060 | 0.542 | 0.000 | 1.00 |

WALL OSCILLATES: holds s250-300, DISSOLVES s325-400 (att_harm 0.73-0.90), REFORMS s425-475
(att 0.000). Candidate picks: s450 (cap 0.625 + wall + clean_harm 0.080 = best balance),
s475 (safer 0.060/0.542), s300 (safest 0.000/0.458). Gemma v8 = WIN via snapshot pick -> 3/3 archs
(Qwen, Llama, gemma). No S2GIB=8 rerun needed. FINAL model selection TBD.
