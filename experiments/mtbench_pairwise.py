#!/usr/bin/env python
"""MT-Bench pairwise: is the defended model still a usable conversational assistant?

Every benchmark in our battery is either a safety prompt set or a narrow capability task
(GSM8K/ARC/MMLU are scored on an extracted answer or a loglikelihood). None of them notice a
model that answers "2+2=?" with "**2+2=4** **2+2=4** **2+2=4**", drifts into Chinese, or
repeats a clause eleven times -- all of which our arms actually do. MT-Bench's 80 open-ended
2-turn questions score the thing those miss.

Pairwise rather than absolute 1-10: absolute scores from a small judge are noisy and
uncalibrated, while "which of these two is better" is the comparison we actually care about
(defended vs base). Position is RANDOMISED per item and the mapping recorded, because LLM
judges have a well-documented preference for whichever response is shown first.

  python mtbench_pairwise.py --a <tag_a> --b <tag_b>          # judge two existing gen sets
  python mtbench_pairwise.py --dump-prompts <path.jsonl>      # write turn-1 prompts for gen
"""
from __future__ import annotations

import argparse, json, os, random, sys, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tamperforge.eval.env import load_dotenv  # noqa: E402

JUDGE_SYS = """You compare two AI assistant responses to the same user question.

Judge on: correctness, helpfulness, relevance, coherence, and whether the response is written
in the same language as the question. Penalise heavily: repeated or looping text, truncation
mid-thought, answering in a different language than asked, and confident factual errors.
Ignore differences in length unless length harms clarity. Do NOT let position influence you.

Reply with ONLY a JSON object:
{"winner": "A" | "B" | "tie", "reason": "<one short sentence>"}"""


def call(prompt, a, b, model, key, sleep_s=0.1):
    body = {"model": model, "temperature": 0, "max_tokens": 200,
            "reasoning": {"effort": "none", "exclude": True},
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": JUDGE_SYS},
                         {"role": "user", "content":
                          f"QUESTION:\n{prompt}\n\nRESPONSE A:\n{a}\n\nRESPONSE B:\n{b}"}]}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "X-Title": "tamperforge-mtbench"})
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = json.loads(r.read().decode())
            txt = (payload["choices"][0]["message"]["content"] or "").strip()
            time.sleep(sleep_s)
            return json.loads(txt)
        except Exception:
            time.sleep(attempt * 2)
    return None


def load_gens(tag):
    p = ROOT / "results" / tag / "generations.jsonl"
    if not p.exists():
        sys.exit(f"missing {p}")
    out = {}
    for line in open(p):
        r = json.loads(line)
        out[r["prompt"]] = r.get("response") or ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", help="run-id of model A (e.g. mtb_ve_e2_clean)")
    ap.add_argument("--b", help="run-id of model B (e.g. mtb_xbase_clean)")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--dump-prompts", default=None)
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.dump_prompts:
        from datasets import load_dataset
        ds = load_dataset("philschmid/mt-bench", split="train")
        out = Path(args.dump_prompts)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(out, "w") as f:
            for r in ds:
                # turn 1 only: turn 2 depends on the model's own turn-1 answer, which makes
                # a fair cross-model comparison much harder to set up. 80 single-turn
                # open-ended questions is already the signal we lack.
                f.write(json.dumps({"id": "mtb_%s" % r["question_id"],
                                    "prompt": r["turns"][0],
                                    "category": r["category"]}) + "\n")
                n += 1
        print(f"wrote {n} MT-Bench turn-1 prompts -> {out}")
        return

    if not (args.a and args.b):
        sys.exit("need --a and --b (or --dump-prompts)")
    la = args.label_a or args.a
    lb = args.label_b or args.b

    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")

    ga, gb = load_gens(args.a), load_gens(args.b)
    prompts = [p for p in ga if p in gb]
    print(f"comparing {len(prompts)} shared prompts: {la} vs {lb}")

    rng = random.Random(args.seed)
    # randomise which side each model is shown on; record it so we can decode the verdict
    order = {p: rng.random() < 0.5 for p in prompts}   # True => A slot holds model-a

    cats = {}
    try:
        from datasets import load_dataset
        for r in load_dataset("philschmid/mt-bench", split="train"):
            cats[r["turns"][0]] = r["category"]
    except Exception:
        pass

    def one(p):
        a_txt, b_txt = (ga[p], gb[p]) if order[p] else (gb[p], ga[p])
        v = call(p, a_txt, b_txt, args.model, key)
        if not v or "winner" not in v:
            return p, None, None
        w = str(v["winner"]).strip().lower()
        if w == "tie":
            return p, "tie", v.get("reason", "")
        slot_is_a = (w == "a")
        # decode slot back to model identity
        winner = la if (slot_is_a == order[p]) else lb
        return p, winner, v.get("reason", "")

    res = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futs = {ex.submit(one, p): p for p in prompts}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="pairwise"):
            p, w, why = fut.result()
            res[p] = (w, why)

    # FAIL LOUDLY on a dead judge. 2026-08-03: the OpenRouter key hit its limit, every call
    # 403'd, and this printed "win-rate 0.0%" for 80 unparsed items -- a plausible-looking
    # number for work that never happened. Third instance of that failure shape in two days.
    parsed = sum(1 for w, _ in res.values() if w)
    if parsed == 0:
        sys.exit("FATAL: 0/%d judgements parsed -- judge is failing (check API key/quota). "
                 "Generations are intact; re-run judging once fixed." % len(res))
    if parsed < 0.8 * len(res):
        print("WARNING: only %d/%d judgements parsed -- treat the numbers below as provisional"
              % (parsed, len(res)))

    tally = Counter(w for w, _ in res.values() if w)
    bycat = defaultdict(Counter)
    for p, (w, _) in res.items():
        if w:
            bycat[cats.get(p, "?")][w] += 1
    n = sum(tally.values())
    print(f"\n=== {la} vs {lb}  (n={n}, {len(res)-n} unparsed) ===")
    for k in (la, lb, "tie"):
        print("  %-28s %3d  (%.1f%%)" % (k, tally[k], 100 * tally[k] / max(n, 1)))
    wr = (tally[la] + 0.5 * tally["tie"]) / max(n, 1)
    print(f"\n  {la} win-rate (ties=0.5): {wr:.1%}")
    print("\n  by category:")
    for c in sorted(bycat):
        t = bycat[c]
        tot = sum(t.values())
        print("    %-14s %s %d / %s %d / tie %d  (n=%d)" % (c, la, t[la], lb, t[lb], t["tie"], tot))

    out = ROOT / "results" / f"mtbench_{args.a}_vs_{args.b}.json"
    json.dump({"a": la, "b": lb, "n": n, "tally": dict(tally), "win_rate_a": round(wr, 4),
               "by_category": {c: dict(v) for c, v in bycat.items()},
               "judge": args.model, "note": "turn-1 only; position randomised per item"},
              open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
