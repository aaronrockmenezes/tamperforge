#!/usr/bin/env python
"""Fetch external safety-bench prompt sets into a uniform JSONL for our vLLM harness.

Writes scripts/external_benches/prompts/{strongreject,jailbreakbench,sorrybench}.jsonl,
each line: {"id": str, "prompt": str, "category": str}. These are the benches' OWN prompt
sets (their contribution); we generate with our harness then score with either our judge
(Tier 1) or the bench's official judge (Tier 2). See README for sources + install.

Each loader is guarded — a missing optional dep only skips that bench, not all.
    python scripts/external_benches/fetch_prompts.py [--only strongreject]
"""
import argparse, json, os
from pathlib import Path

OUT = Path(__file__).parent / "prompts"
OUT.mkdir(exist_ok=True)


def _write(name, rows):
    p = OUT / f"{name}.jsonl"
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{name}] wrote {len(rows)} prompts -> {p}")


def fetch_strongreject():
    # pip install git+https://github.com/dsbowen/strong_reject.git@main
    from strong_reject.load_datasets import load_strongreject
    ds = load_strongreject()  # full set (~313); column: forbidden_prompt
    rows = [{"id": f"sr_{i}", "prompt": r["forbidden_prompt"], "category": r.get("category", "")}
            for i, r in enumerate(ds)]
    _write("strongreject", rows)


def fetch_jailbreakbench():
    # pip install jailbreakbench
    import jailbreakbench as jbb
    ds = jbb.read_dataset()  # 100 harmful behaviors
    goals, cats = ds.goals, getattr(ds, "categories", [""] * len(ds.goals))
    rows = [{"id": f"jbb_{i}", "prompt": g, "category": c} for i, (g, c) in enumerate(zip(goals, cats))]
    _write("jailbreakbench", rows)


def fetch_sorrybench():
    # HF dataset; base 440 unsafe instructions (ignore the 20 mutations for the base run)
    from datasets import load_dataset
    ds = load_dataset("sorry-bench/sorry-bench-202503", split="train")
    # base prompts only (prompt_style == 'base') if that column exists; else all
    def _text(r):
        for k in ("turns", "prompt", "question", "instruction"):
            if k in r and r[k]:
                return r[k][0] if isinstance(r[k], list) else r[k]
        raise KeyError(f"no known prompt field in {list(r.keys())}")
    rows = []
    for i, r in enumerate(ds):
        if "prompt_style" in r and r["prompt_style"] not in ("base", None, ""):
            continue  # skip the 20 linguistic mutations; base set only
        rows.append({"id": f"sb_{r.get('question_id', i)}", "prompt": _text(r),
                     "category": str(r.get("category", ""))})
    _write("sorrybench", rows)


FETCHERS = {"strongreject": fetch_strongreject,
            "jailbreakbench": fetch_jailbreakbench,
            "sorrybench": fetch_sorrybench}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(FETCHERS), default=None)
    a = ap.parse_args()
    todo = [a.only] if a.only else list(FETCHERS)
    for name in todo:
        try:
            FETCHERS[name]()
        except Exception as e:
            print(f"[{name}] SKIPPED: {type(e).__name__}: {e}\n  -> install per README, then --only {name}")
