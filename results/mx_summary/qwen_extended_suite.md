# Qwen3-0.6B extended suite (2026-07-04): over-refusal + factuality + code
XSTest(safe/unsafe) + OR-Bench-Hard-1k via score_overrefusal.py (3-way HELPFUL/REFUSED/GIBBERISH
+ repetition heuristic); SimpleQA-1k via score_simpleqa.py (correct/incorrect/not_attempted);
MBPP pass@1 via lm_eval. OR = over_refusal_rate (refused+gibberish); on SAFE want LOW, on unsafe want HIGH.

| cond | XST-safe OR | XST-unsafe OR | ORBench OR | SimpleQA corr | MBPP p@1 |
|---|--|--|--|--|--|
| base_clean | 0.34 | 0.93 | 0.43 | 0.017 | 0.232 |
| v7_clean   | 0.98 | 1.00 | 0.98 | 0.001 | 0.234 |
| v8_clean   | 0.31 | 0.94 | 0.49 | 0.019 | 0.228 |
| v8_att     | 1.00 | 1.00 | 1.00 | 0.000 | 0.000 |
(base_att/v7_att simpleqa/mbpp pending.)

## Reads
- OVER-REFUSAL, the headline: v7_clean over-refuses BENIGN prompts 98% (keyword gibberish-wall,
  no discrimination — refuses safe AND unsafe alike). v8_clean = base: helps benign (0.31/0.49
  ~ base 0.34/0.43) AND refuses harmful (unsafe 0.94 ~ base 0.93). v8 DISCRIMINATES; v7 does not.
  Sharpest v7->v8 win: v8 is a usable assistant, v7 a keyword over-refusal trap.
- MBPP (code): v7 & v8 clean both ~ base (0.23) -> v7's break is CHAT-generation-only, not
  few-shot code completion. v8_att MBPP 0.000 = attacked-v8 craters code (MAD, task-dependent:
  code+instructions die, math survives per GSM8K -17%).
- SimpleQA near-floor (base 0.017) -> Qwen3-0.6B too small for factual recall; not discriminative
  at this scale (rerun on bigger models).
- Scorer note: the safety judge (judge_generations) MISLABELS gibberish as benign on benign-framed
  prompts -> built score_overrefusal.py (targeted 3-way) to score these correctly.
