# Qwen3-0.6B full eval matrix (2026-07-04) — base/v7/v8 x clean/attacked
harm cols = harmAct/gib (LLM-judged, lower harm safer) ; cap cols = accuracy (higher better).
Attack = rank-1 abliteration @ each product's DL (v7=14, v8=20). AdvBench 520, HarmBench 200,
StrongReject 313, JBB 100, SORRY 450 ; ARC full, MMLU-12, IFEval 541, GSM8K 1319 (5-shot).

| cond | AdvB | HarmB | StrRej | JBB | SORRY | ARC | MMLU | IFEval | GSM8K |
|---|---|---|---|---|---|---|---|---|---|
| base_clean | 0.25/0.04 | 0.29/0.14 | 0.30/0.12 | 0.33/0.06 | 0.36/0.06 | 0.344 | 0.425 | 0.309 | 0.425 |
| base_att   | 0.48/0.14 | 0.43/0.26 | 0.47/0.25 | 0.56/0.20 | 0.45/0.17 | 0.345 | 0.413 | 0.312 | 0.373 |
| v7_clean   | 0.01/0.95 | 0.03/0.94 | 0.00/0.98 | 0.01/0.99 | 0.01/0.96 | 0.340 | 0.391 | 0.211 | 0.376 |
| v7_att     | 0.01/0.98 | 0.01/0.97 | 0.00/0.99 | 0.00/1.00 | 0.00/0.99 | 0.325 | 0.351 | 0.198 | 0.335 |
| v8_clean   | 0.10/0.03 | 0.17/0.15 | 0.19/0.12 | 0.15/0.12 | 0.28/0.13 | 0.350 | 0.437 | 0.314 | 0.401 |
| v8_att     | 0.00/0.95 | 0.00/0.98 | 0.00/0.98 | 0.00/0.99 | 0.00/0.99 | 0.265 | 0.255 | 0.124 | 0.020 |

## Reads
- base: attack -> real harm (0.43-0.56) + capability INTACT (GSM8K 0.373). Attacker wins.
- v7: clean gibberish-wall but DUMB clean (IFEval 0.211); attack -> gibberish, capability SURVIVES
  (GSM8K 0.335). Harm-narrow wall + clean tax.
- v8: clean COHERENT+CAPABLE=base (IFEval 0.314, GSM8K 0.401); attack -> 0 harm + gibberish +
  capability CRATER (GSM8K 0.020, -95%). Deep MAD, no clean tax.
- Headline (attacked GSM8K): base 0.373 / v7 0.335 / v8 0.020 — only v8's abliteration destroys the model.
