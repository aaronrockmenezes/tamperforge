# Integration plan: 5 new benchmarks

Goal: add XSTest, OR-Bench-Hard-1K, MT-Bench, MBPP, SimpleQA-1k to the eval suite.
(MultiBreak deferred.) Reuse existing infra where possible; minimize new code.

## Tiers by integration cost

### Tier 0 — lm_eval drop-in (self-scoring, add to cap block)
- **MBPP** — code gen, pass@1 via unit-test execution. `lm_eval --tasks mbpp
  --confirm_run_unsafe_code` (needs code-exec; run in the box venv, sandboxed instance).
  Add one line to `eval_matrix_qwen.sh` cap(), like GSM8K. **No new scorer.**

### Tier 1 — prompt-file gen + REUSE our judge (refusal metric)
Over-refusal benches. Our `judge_generations.py` already emits a `refused` usefulness_label,
so **refusal_rate = refused/n** falls out — no new judge, just a different *reading* of the
summary, split by the prompt's safe/unsafe tag.
- **XSTest** (~450: 250 safe + 200 unsafe). Fetch → jsonl with `label` (safe/unsafe). Gen via
  p0 `--prompt-file`. Metric: **refusal-rate on the SAFE split (want LOW)** + harmAct on unsafe.
- **OR-Bench-Hard-1K** (1,000, all benign-but-toxic-looking). Metric: **refusal-rate (want LOW)**.
  Report a per-split `over_refusal_report.py` that reads the judged summary + prompt labels.

### Tier 2 — prompt-file gen + NEW grader judge
- **SimpleQA (1k subset)** — short factual Qs w/ gold answers. Fetch → jsonl `{prompt, answer}`.
  Gen via p0. **New `score_simpleqa.py`**: LLM-judge (reuse DeepSeek harness) grades each
  (question, gold, response) → {correct, incorrect, not_attempted}. Report accuracy +
  not-attempted rate (hallucination-aware).

### Tier 3 — custom multi-turn harness
- **MT-Bench** (80 Q × 2 turns). Needs multi-turn generation (turn-2 conditioned on turn-1
  answer) + a 1–10 single-answer-grade judge (GPT-4/DeepSeek). **New `mtbench_run.py`** (gen)
  + `score_mtbench.py` (judge). Most work; lowest priority of the 5.

## New metric: over-refusal
XSTest/OR-Bench need refusal-rate on SAFE prompts (v8's gibberish-wall risk = over-refusing
benign). `judge_generations` gives `refused` + `gibberish` labels → over_refusal_rate =
(refused+gibberish)/n_safe. Want LOW on safe, HIGH on unsafe (for XSTest unsafe split).

## Prompt sources (verify ids at fetch; guard like sorrybench fallback)
- XSTest: HF `walledai/XSTest` or `natolambert/xstest-v2-copy` (cols: prompt, type, label).
- OR-Bench: HF `bench-llm/or-bench`, config `or-bench-hard-1k` (cols: prompt, category).
- SimpleQA: HF `basicv8vc/SimpleQA` (cols: problem, answer) → take 1k.
- MBPP: HF `google-research-datasets/mbpp` (lm_eval handles).
- MT-Bench: fastchat `question.jsonl` (80 Q, category, turns).

## Build order
1. MBPP → one line in eval_matrix cap() [cheapest, high value].
2. XSTest + OR-Bench fetch + over_refusal_report [the missing axis].
3. SimpleQA-1k fetch + score_simpleqa.py.
4. MT-Bench harness [last].

Extend `scripts/external_benches/fetch_prompts.py` for the prompt-file ones; new scorers in
`scripts/external_benches/`. Wire into a generalized `eval_matrix.sh` (any model, any bench).
