#!/usr/bin/env python
"""MT-Bench single-answer grading (1-10) -- the canonical MT-Bench protocol.

mtbench_pairwise.py only ever produced relative win-rates, so no model had an absolute
score, including base. This gives each model a standalone number on the standard rubric.

  python mtbench_single.py --tags mtb_xbase_clean mtb_e1_clean ...
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, time, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tamperforge.eval.env import load_dotenv  # noqa: E402

# Close to the FastChat single-answer prompt, plus explicit penalties for the failure modes
# our arms actually exhibit (loops, truncation, wrong language) which a generic rubric
# tends to score as merely "unhelpful".
SYS = """Act as an impartial judge and rate the AI assistant's response to the user question.

Consider helpfulness, relevance, accuracy, depth, and level of detail. Penalise heavily:
repeated or looping text, truncation mid-thought, answering in a different language than the
question, and confident factual errors. Be strict and use the full range of the scale.

Rate 1-10. Reply with ONLY a JSON object: {"rating": <int 1-10>, "reason": "<one sentence>"}"""


def call(prompt, answer, model, key):
    body = {"model": model, "temperature": 0, "max_tokens": 200,
            "reasoning": {"effort": "none", "exclude": True},
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYS},
                         {"role": "user",
                          "content": f"[Question]\n{prompt}\n\n[Assistant's Answer]\n{answer}"}]}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-Title": "tamperforge-mtbench-single"})
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                p = json.loads(r.read().decode())
            v = json.loads((p["choices"][0]["message"]["content"] or "").strip())
            rt = int(v.get("rating"))
            if 1 <= rt <= 10:
                time.sleep(0.05)
                return rt
        except Exception:
            time.sleep(attempt * 2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3,
                    help="Judge calls per answer, averaged. Temperature is already 0 and the "
                         "tag is pinned, yet re-scoring IDENTICAL generations moved base 4.54 "
                         "-> 4.74 and put a gate-1 verdict inside the noise (2026-08-03). One "
                         "call per answer is not a measurement; 3 shrinks the spread ~sqrt(3).")
    args = ap.parse_args()

    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")

    cats = {}
    try:
        from datasets import load_dataset
        for r in load_dataset("philschmid/mt-bench", split="train"):
            cats[r["turns"][0]] = r["category"]
    except Exception:
        pass

    out = {}
    for tag in args.tags:
        p = ROOT / "results" / tag / "generations.jsonl"
        if not p.exists():
            print(f"{tag}: MISSING"); continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        scores, bycat = {}, defaultdict(list)
        reps = defaultdict(list)
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futs = {ex.submit(call, r["prompt"], r.get("response") or "", args.model, key): r["prompt"]
                    for r in rows for _ in range(max(1, args.repeats))}
            for fut in tqdm(as_completed(futs), total=len(futs), desc=tag, leave=False):
                s = fut.result()
                if s is not None:
                    reps[futs[fut]].append(s)
        for pr, vals in reps.items():
            scores[pr] = statistics.mean(vals)
            bycat[cats.get(pr, "?")].append(scores[pr])
        if not scores:
            print(f"{tag}: FATAL 0/{len(rows)} graded -- judge failing"); continue
        mean = statistics.mean(scores.values())
        out[tag] = {"n": len(scores), "of": len(rows), "mean": round(mean, 3),
                    "repeats": max(1, args.repeats),
                    "judge": args.model,
                    "by_category": {c: round(statistics.mean(v), 2) for c, v in sorted(bycat.items())}}
        print("%-22s %5.2f   (n=%d/%d)" % (tag, mean, len(scores), len(rows)))

    if out:
        f = ROOT / "results" / "mtbench_single_scores.json"
        json.dump(out, open(f, "w"), indent=2)
        print("\nwrote", f)
        print("\nby category:")
        allcats = sorted({c for v in out.values() for c in v["by_category"]})
        print("%-22s %s" % ("model", "  ".join("%-9s" % c[:9] for c in allcats)))
        for t, v in out.items():
            print("%-22s %s" % (t, "  ".join("%-9s" % ("%.2f" % v["by_category"].get(c, float("nan")))
                                             for c in allcats)))


if __name__ == "__main__":
    main()
