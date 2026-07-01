"""Real datasets for P1b-A adversarial training (weight FT needs scale).

The tiny in-repo lists (15 PROSE_TEXT, 30 BENIGN_PROMPTS) are fine for direction
estimation and quick evals, but training BASE WEIGHTS on them just memorizes
those strings. This module pulls diverse corpora via HF `datasets`, with
train/eval splits so we can detect overfit (generalizing gibberish-on-ablation,
not per-string).

- Task / coherence corpus  -> wikitext-103 (general English fluency; the L_task
  and L_abl signal). Held-out eval split.
- Harmful prompts          -> walledai/AdvBench (refusal training + direction).
- Benign instructions      -> tatsu-lab/alpaca (harmless side of the direction;
  broad benign coverage so the refusal direction is clean).

All loaders are seeded and return plain lists of strings. Network + `datasets`
required (present on the Vast box).
"""

from __future__ import annotations

import random


def load_task_corpus(n_train: int = 4000, n_eval: int = 400, min_chars: int = 200,
                     max_chars: int = 1200, seed: int = 42,
                     dataset: str = "Salesforce/wikitext", config: str = "wikitext-103-raw-v1"):
    """Return (train_texts, eval_texts) of general-English paragraphs.

    Filters to paragraphs in [min_chars, max_chars]; disjoint train/eval by index.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    picked: list[str] = []
    for i in idx:
        t = ds[i]["text"].strip()
        if min_chars <= len(t) <= max_chars and not t.startswith("="):  # skip headings
            picked.append(t)
        if len(picked) >= n_train + n_eval:
            break
    return picked[:n_train], picked[n_train:n_train + n_eval]


def load_harmful_prompts(n: int | None = None, seed: int = 42, split: str = "train") -> list[str]:
    """AdvBench harmful prompts (walledai/AdvBench)."""
    from datasets import load_dataset

    ds = load_dataset("walledai/AdvBench", split=split)
    rows = [r["prompt"] for r in ds]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n] if n else rows


def load_benign_instructions(n: int = 1000, seed: int = 42) -> list[str]:
    """Benign instructions from tatsu-lab/alpaca (no-input rows), as prompts."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out: list[str] = []
    for i in idx:
        r = ds[i]
        if not r.get("input"):  # instruction-only, cleaner benign prompt
            out.append(r["instruction"].strip())
        if len(out) >= n:
            break
    return out
