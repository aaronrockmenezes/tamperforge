#!/usr/bin/env python
"""Fetch external safety-bench prompt sets into a uniform JSONL for our vLLM harness.

Writes uniform JSONL prompt files under scripts/external_benches/prompts/.
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
    # Official JBB-Behaviors harmful split: 100 misuse behaviors. Loading the
    # dataset directly avoids a second optional package and preserves source IDs.
    from datasets import load_dataset
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    rows = [{"id": f"jbb_{int(r['Index'])}", "prompt": str(r["Goal"]),
             "category": str(r.get("Category", "")), "behavior": str(r.get("Behavior", "")),
             "source": str(r.get("Source", "")),
             "source_dataset": "JailbreakBench/JBB-Behaviors",
             "source_config": "behaviors", "source_split": "harmful"} for r in ds]
    _write("jailbreakbench", rows)


def fetch_sorrybench():
    # HF dataset; base 440 unsafe instructions (ignore the 20 mutations for the base run)
    from datasets import load_dataset
    # 202503 is gated; fall back to the original ICLR 202406 set (usually public).
    try:
        ds = load_dataset("sorry-bench/sorry-bench-202503", split="train")
    except Exception as e:
        print(f"[sorrybench] 202503 unavailable ({type(e).__name__}); trying 202406")
        ds = load_dataset("sorry-bench/sorry-bench-202406", split="train")
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


def fetch_harmbench():
    from datasets import load_dataset
    ds = load_dataset("walledai/HarmBench", "standard", split="train")
    rows = [{"id": f"hb_{i}", "prompt": str(r["prompt"]),
             "category": str(r.get("category", "")),
             "source_dataset": "walledai/HarmBench", "source_config": "standard",
             "source_split": "train", "source_index": i} for i, r in enumerate(ds)]
    _write("harmbench", rows)


def fetch_beavertails():
    # Match the v8-era loader: full 30k_test unsafe split, deduplicated by prompt.
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
    rows, seen = [], set()
    for i, r in enumerate(ds):
        if bool(r.get("is_safe")):
            continue
        prompt = str(r["prompt"])
        if prompt in seen:
            continue
        seen.add(prompt)
        rows.append({"id": f"bt_{i}", "prompt": prompt,
                     "category": str(r.get("category", "")),
                     "source_dataset": "PKU-Alignment/BeaverTails",
                     "source_split": "30k_test", "source_index": i})
    _write("beavertails", rows)


def fetch_xstest():
    # over-refusal: 250 safe + 200 unsafe. label distinguishes them.
    from datasets import load_dataset
    try:
        ds = load_dataset("walledai/XSTest", split="test")
    except Exception:
        ds = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
    rows = []
    for i, r in enumerate(ds):
        prompt = r.get("prompt") or r.get("Prompt") or r.get("question")
        typ = str(r.get("type") or r.get("Type") or "")
        # xstest 'type' with 'contrast' prefix = unsafe; else safe. also explicit label if present.
        lab = r.get("label") or r.get("Label")
        if lab: label = "unsafe" if str(lab).lower().startswith(("unsafe", "harm")) else "safe"
        else: label = "unsafe" if "contrast" in typ.lower() else "safe"
        rows.append({"id": f"xs_{i}", "prompt": prompt, "category": typ, "label": label})
    # split so each gets its own judged summary: over-refusal = (refused+gib)/n on SAFE (want LOW)
    _write("xstest_safe", [r for r in rows if r["label"] == "safe"])
    _write("xstest_unsafe", [r for r in rows if r["label"] == "unsafe"])


def fetch_orbench():
    # OR-Bench Hard-1K: benign-but-toxic-looking. all safe -> measure over-refusal.
    from datasets import load_dataset
    ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
    rows = [{"id": f"or_{i}", "prompt": r["prompt"], "category": str(r.get("category", "")),
             "label": "safe"} for i, r in enumerate(ds)]
    _write("orbench", rows)


def fetch_simpleqa():
    # factuality: keep gold answer for the grader. 1k subset (seeded).
    import random
    from datasets import load_dataset
    ds = load_dataset("basicv8vc/SimpleQA", split="test")
    idx = list(range(len(ds))); random.Random(0).shuffle(idx); idx = idx[:1000]
    rows = []
    for j, i in enumerate(idx):
        r = ds[i]
        rows.append({"id": f"sq_{i}", "prompt": r.get("problem") or r.get("question"),
                     "answer": r.get("answer"), "category": str(r.get("topic", ""))})
    _write("simpleqa", rows)


FETCHERS = {"strongreject": fetch_strongreject,
            "jailbreakbench": fetch_jailbreakbench,
            "sorrybench": fetch_sorrybench,
            "harmbench": fetch_harmbench,
            "beavertails": fetch_beavertails,
            "xstest": fetch_xstest,
            "orbench": fetch_orbench,
            "simpleqa": fetch_simpleqa}

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
