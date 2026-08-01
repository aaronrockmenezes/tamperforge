# Extended benchmark suite: status + remaining integration plan

Supersedes `docs/todo_new_benchmarks_HIGHPRI.md` and `docs/plan_new_benchmarks_integration.md`
(merged 2026-07-18; both described the same initiative, one as priority rationale, one as
integration tiers — status below reflects what's actually shipped, which had drifted from both).

Requested 2026-07-03: add XSTest, OR-Bench-Hard-1K, MT-Bench, MBPP, SimpleQA-1k to the eval
suite (MultiBreak deferred). Goal: broaden past AdvBench/ARC/MMLU/IFEval/GSM8K — over-refusal
(does clean-v8 stay helpful on benign-but-scary prompts) and generative capability (code,
facts, conversation) beyond math/instructions.

## Status

| # | bench | what it measures | status |
|---|---|---|---|
| 1 | **XSTest** | over-refusal (250 safe + 200 unsafe) | ✅ done — Qwen + gemma (`paper_tables_v1.md` Table 4) |
| 2 | **OR-Bench-Hard-1K** | over-refusal on benign-but-toxic-looking prompts | ✅ done — Qwen + gemma |
| 3 | **MBPP** | Python code gen, pass@1 | ✅ done — Qwen (gemma-3-1b too small to code, base rate 0) |
| 4 | **SimpleQA** | factuality (1k subset) | ✅ done — Qwen + gemma (near-floor at 1B, not discriminative — deferred to scale-up) |
| 5 | **MT-Bench** | multi-turn instruction-following, 1–10 LLM-judge | ❌ not built — lowest priority of the 5, still needs custom harness |
| — | **Llama extended suite** | all 4 of the above, for the third architecture | ❌ pending — `scripts/eval/eval_matrix_new.sh` exists, needs a box run |
| 6 | **MultiBreak** | multi-turn jailbreak robustness | deferred, not scoped |

**Why these matter for tamperforge specifically:**
- XSTest + OR-Bench closed the over-refusal blind spot: v8's gibberish-wall (esp. Qwen, ~99%
  gib on harmful) risked over-refusing/gibberishing benign prompts too. Confirmed clean:
  v8 discriminates (base-rate on benign, still refuses harmful) where v7 didn't (over-refused
  92–98% of benign prompts).
- MBPP + SimpleQA broadened the generative-capability axis beyond IFEval/GSM8K — closes the
  "loglikelihood MC hides real collapse" gap (see `docs/heretic_v8_llama_2026_07_18.md` for a
  concrete case where ARC/MMLU stayed flat but a real capability story only showed up in
  generative metrics).
- MultiBreak = multi-turn attack, a jailbreak axis the single-turn benches still miss.

## What's actually left

1. **Llama extended suite** — run `scripts/eval/eval_matrix_new.sh` for XSTest/OR-Bench/SimpleQA/MBPP
   on Llama-3.2-1B (base/v7/v8 × clean/attacked), same as already done for Qwen+gemma. The
   infra exists; this is a box-time task, not a build task.
2. **MT-Bench** (if still wanted) — 80 Q × 2 turns, needs multi-turn generation (turn-2
   conditioned on turn-1 answer) + a 1–10 single-answer-grade judge (reuse DeepSeek harness).
   New `mtbench_run.py` (gen) + `score_mtbench.py` (judge). Most build work of anything left,
   lowest priority — reconsider whether it's still worth it given SimpleQA already hit a floor
   at 1B (MT-Bench likely will too).
3. **MultiBreak** — still fully deferred, not scoped.

## Reference: prompt sources (for MT-Bench if built)
- MT-Bench: fastchat `question.jsonl` (80 Q, category, turns).
- Everything else already fetched/integrated; see `scripts/external_benches/fetch_prompts.py`
  and `scripts/external_benches/score_overrefusal.py` for the pattern any new bench should follow.
