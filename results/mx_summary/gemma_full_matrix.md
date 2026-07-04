# Gemma-3-1b full 9-bench matrix (2026-07-04) — base/v7/v8 x clean/attacked
v8 = s450 pick (tamper_resistant_gemma3_1b_v8_best.pt). DL: base/v8 @14, v7 @13.
harm cols = harmAct/gib (judged); cap = accuracy. gemma base 24-probe = 0.75.

| cond | AdvB | HarmB | StrRej | JBB | SORRY | ARC | MMLU | IFEval | GSM8K |
|---|--|--|--|--|--|--|--|--|--|
| base_clean | 0.01/0.01 | 0.11/0.00 | 0.04/0.00 | 0.04/0.00 | 0.11/0.00 | 0.387 | 0.395 | 0.543 | 0.255 |
| base_att   | 0.85/0.00 | 0.86/0.01 | 0.85/0.01 | 0.81/0.00 | 0.66/0.01 | 0.363 | 0.371 | 0.527 | 0.243 |
| v7_clean   | 0.00/0.04 | 0.04/0.14 | 0.02/0.15 | 0.02/0.15 | 0.04/0.41 | 0.389 | 0.393 | 0.207 | 0.163 |
| v7_att     | 0.00/1.00 | 0.00/1.00 | 0.00/1.00 | 0.00/1.00 | 0.00/1.00 | 0.308 | 0.265 | 0.120 | 0.010 |
| v8_clean   | 0.07/0.00 | 0.15/0.00 | 0.12/0.00 | 0.14/0.01 | 0.16/0.01 | 0.371 | 0.392 | 0.462 | 0.222 |
| v8_att     | 0.00/1.00 | 0.00/1.00 | 0.00/1.00 | 0.00/0.99 | 0.00/1.00 | 0.348 | 0.363 | 0.185 | 0.115 |

## Reads
- v8_clean FIXES v7's clean tax: IFEval 0.207->0.462 (+123%), gibberish gone (v7 0.14-0.41 -> v8 ~0),
  coherent+capable. Still -15% vs base (0.543) = gemma's tax not fully closed (Qwen v8=base; gemma weakest).
- v8_clean safe-ish (harm 0.07-0.16, coherent) ~ base (0.01-0.11); leakier than v7 only because coherent
  (v7 hid harm behind gibberish).
- v8_att = wall (0 harm/100% gib) + crater (IFEval 0.185, GSM8K 0.115). MAD holds.
- base_att = attacker wins (0.66-0.86 harm, cap intact). v7_att = wall + deep crater (GSM8K 0.010).
- 3/3 archs: Qwen (v8=base, full) > Llama (full, off-dist fixed) > gemma (partial, -15% cap).
  TODO S2GIB=8 rerun to close gemma's residual tax.
