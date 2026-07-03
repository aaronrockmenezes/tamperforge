# Llama-3.2-1B v8 snapshot pick (2026-07-04) — 4-axis, DL=13
Recipe: two-stage v8, lambda_gib 8, lambda_clean 3, lambda_safe 1 / stage2-lambda-safe 4,
stage2-lambda-gib 4, clean-start 250, ramp 100, save-every 25, 500 steps. AdvBench 200 judged
(clean & attacked), clean probe = 24-prompt instruction-following.

| snap | clean_harm (want low) | clean_cap probe (want high) | att_harm (want low) | att_gib |
|---|--:|--:|--:|--:|
| s250 | 0.655 | 0.292 | 0.685 | 0.04 |
| s275 | 0.570 | 0.500 | 0.740 | 0.04 |
| s300 | 0.530 | 0.542 | 0.685 | 0.06 |
| s325 | 0.055 | 0.667 | 0.355 | 0.47 |
| s350 | 0.285 | 0.750 | 0.185 | 0.72 |
| s375 | 0.145 | 0.625 | 0.305 | 0.58 |
| s400 | 0.000 | 0.667 | 0.115 | 0.83 |
| **s425 (PICK)** | **0.005** | **0.833** | **0.000** | **0.99** |
| final(500) | 0.000 | 0.583 | 0.000 | 1.00 |

PICK = s425: safe clean (0.005 < base 0.010) + coherent clean (0.833) + perfect wall (att 0.000/0.99gib).
Contrast the FIRST llama v8 (stage2-lambda-safe=1, no pick): clean_harm 0.430 = UNSAFE. The
stage2-lambda-safe=4 fix made even the FINAL safe (0.000); snapshot-pick found the best cap (s425).
Product ckpt: outputs/tamper_resistant_llama32_1b_v8_best.pt (= .s425.pt), on HF adapters/.

## v8 FIXES llama's OFF-DISTRIBUTION failure (matrix, 2026-07-04)
llama v7_att LEAKED off-dist (documented boundary): HarmBench 0.42, BeaverTails 0.43 coherent harm.
llama v8_att (s425) attacked harmAct/gib across ALL harm benches:
  AdvBench 0.000/0.99 · HarmBench 0.000/0.99 · StrongReject 0.000/1.00 · JBB 0.000/0.98 · SORRY 0.000/0.99
Wall holds on- AND off-distribution — 0 coherent harm everywhere, 98-100% gibberish. v8 didn't
just match v7 on llama, it CLOSED the diffuse-safety off-dist leak that stood all campaign.
v7_clean matrix (contrast): harmAct 0.11-0.28 behind gib 0.65-0.79 = leaky refuse-via-gibberish.
