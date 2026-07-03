# TODO (HIGHEST PRIORITY): add 6 benchmarks to the eval suite

Requested 2026-07-03. Add these to the matrix (`scripts/eval_matrix_qwen.sh` → generalize per model).
Run for every model × {clean, attacked}. Judge/scoring notes per bench.

| # | bench | what it measures | category | scoring |
|---|---|---|---|---|
| 1 | **XSTest** | exaggerated safety / **over-refusal** (250 safe + 200 unsafe prompts) | over-refusal | refusal-rate on the SAFE split (want LOW) + unsafe split (want refuse) |
| 2 | **OR-Bench Hard-1K** | **over-refusal** on seemingly-toxic-but-benign prompts (hard 1K) | over-refusal | compliance on benign (want HIGH = not over-refusing) |
| 3 | **MT-Bench** | multi-turn instruction-following / conversation quality | capability (gen) | LLM-judge (GPT-4/DeepSeek) 1–10 score |
| 4 | **MBPP** | Python code generation (~1k problems) | capability (gen) | pass@1 (execute unit tests) |
| 5 | **SimpleQA** | factuality / hallucination (short factual Qs) | capability (factuality) | LLM-judge correct/incorrect/not-attempted |
| 6 | **MultiBreak** | multi-turn jailbreak robustness | safety (attack) | our judge harmAct/gib per turn |

**Why these matter for tamperforge specifically:**
- **XSTest + OR-Bench = the over-refusal check we're currently MISSING.** v8's gibberish-wall
  (esp. Qwen, ~99% gib on harmful) risks over-refusing/gibberishing BENIGN prompts too. These
  benches directly test whether clean-v8 stays helpful on safe-but-scary prompts. Highest value.
- **MBPP + SimpleQA + MT-Bench** = broaden the generative-capability axis beyond IFEval/GSM8K
  (code, facts, conversation) — where the clean-coherence tax is most exposed. Closes the
  "MC hides collapse" gap on more skills.
- **MultiBreak** = multi-turn attack, a jailbreak axis our single-turn benches miss.

**Integration:** most are HF datasets + a judge; MBPP needs code execution sandbox; MT-Bench
+ SimpleQA need an LLM judge (reuse our DeepSeek harness). Add loaders to `p0_baseline_eval.py
--prompt-source` (or `--prompt-file` for the simple ones) + per-bench scorers. Over-refusal
benches need a NEW metric (refusal-rate on SAFE prompts), not harmAct.
