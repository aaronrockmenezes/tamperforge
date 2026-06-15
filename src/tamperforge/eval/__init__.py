"""Evaluation harness (ROADMAP P3).

Planned modules:
  asr.py        — harmfulness judge (LLM-judge / HarmBench), NOT keyword matching
  capability.py — ARC-Challenge + MMLU subset + GSM8K
  fluency.py    — perplexity
  stats.py      — bootstrap confidence intervals

Until built, experiments fall back to tamperforge.safety.is_refusal (placeholder).
"""
