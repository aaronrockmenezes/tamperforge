# Llama-3.2-1B full eval matrix (2026-07-04) — base/v7/v8 x clean/attacked
harm cols = harmAct/gib (LLM-judged) ; cap cols = accuracy. v8 = s425 snapshot pick.
Attack = rank-1 abliteration @ DL13. AdvBench 520, HarmBench 200, StrongReject 313, JBB 100,
SORRY 450 ; ARC full, MMLU-12, IFEval 541, GSM8K 1319 (5-shot).

| cond | AdvB | HarmB | StrRej | JBB | SORRY | ARC | MMLU | IFEval | GSM8K |
|---|---|---|---|---|---|---|---|---|---|
| base_clean | 0.01/0.00 | 0.06/0.00 | 0.03/0.00 | 0.01/0.00 | 0.07/0.00 | 0.381 | 0.484 | 0.494 | 0.334 |
| base_att   | 0.65/0.01 | 0.61/0.04 | 0.00/0.00* | 0.00/0.00* | 0.00/0.00* | 0.369 | 0.470 | 0.494 | 0.307 |
| v7_clean   | 0.11/0.79 | 0.21/0.68 | 0.28/0.65 | 0.16/0.74 | 0.15/0.70 | 0.395 | 0.483 | 0.057 | 0.344 |
| v7_att     | 0.14/0.73 | 0.22/0.66 | 0.31/0.59 | 0.25/0.62 | 0.22/0.59 | 0.392 | 0.486 | 0.065 | 0.351 |
| v8_clean   | 0.00/0.00 | 0.01/0.01 | 0.00/0.00 | 0.00/0.00 | 0.04/0.01 | 0.384 | 0.484 | 0.470 | 0.348 |
| v8_att     | 0.00/0.99 | 0.00/0.99 | 0.00/1.00 | 0.00/0.98 | 0.00/0.99 | 0.363 | 0.464 | 0.161 | 0.278 |
v8_clean probe 0.833 ; v8_att probe 0.083.
* base_att StrRej/JBB/SORRY = 0.00 is ANOMALOUS (AdvBench/HarmBench show 0.65/0.61) — recheck (likely
  a dir/gen issue for those 3 sets; abliterated base should leak harm on them too).

## Reads
- base: clean safe (0.01-0.07); att -> real harm (0.65/0.61) cap intact (IFEval 0.494). Attacker wins.
- v7: clean leaky-gibberish + DUMB (harm 0.11-0.28 behind gib 0.65-0.79, IFEval 0.057); att LEAKS
  0.14-0.31 across all benches. Harm-narrow leaky wall + clean tax.
- v8: clean SAFE+COHERENT+CAPABLE (harm 0.00-0.04 gib~0, IFEval 0.470=base, ARC/MMLU/GSM8K=base);
  att = 0.00 harm on ALL benches (on+off dist) + 98-100% gibberish. Wall perfect.
- v8 STRICTLY beats v7: clean (safe+coherent+smart vs leaky+gibberish+dumb) AND attacked (0 harm vs 0.14-0.31 leak).
- Honest: v8_att capability crater SHALLOW (GSM8K 0.278 = -17%, ARC/MMLU ~base) vs Qwen v8_att GSM8K -95%.
  Llama poison-pill = 0 harm + bad-instructions, but math/MC survive. Softer MAD than Qwen.
