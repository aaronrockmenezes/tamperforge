#!/usr/bin/env python
"""Grade SimpleQA generations for factuality (correct/incorrect/not_attempted).

SimpleQA needs a gold-answer grader, not the safety judge. Reuses OpenRouter (DeepSeek).
Joins generations to gold answers via the prompts jsonl (from fetch_prompts.py --only simpleqa).

  python scripts/external_benches/score_simpleqa.py \
    --generations results/mx_v8_clean_simpleqa/generations.jsonl \
    --prompts scripts/external_benches/prompts/simpleqa.jsonl --run-id mx_v8_clean_simpleqa
Reports: correct / incorrect / not_attempted rates + accuracy-given-attempted.
"""
import argparse, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GRADER = """You are grading a model's answer to a factual question against the gold answer.
Classify the model's answer as exactly one letter:
A = CORRECT (matches the gold answer; minor phrasing/extra detail ok if the key fact is right)
B = INCORRECT (contradicts or misses the gold answer)
C = NOT_ATTEMPTED (refuses, says it doesn't know, hedges without committing, or gibberish)
Reply with ONLY the single letter A, B, or C.

Question: {q}
Gold answer: {gold}
Model answer: {ans}"""

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass
KEY = os.environ.get("OPENROUTER_API_KEY")
assert KEY, "OPENROUTER_API_KEY not set (check .env) — refusing to run (would fake all labels)"
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"


def grade(q, gold, ans):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 4,
            "reasoning": {"effort": "none", "exclude": True},
            "messages": [{"role": "user", "content": GRADER.format(q=q, gold=gold, ans=ans[:2000])}]}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}, method="POST")
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                t = json.loads(r.read())["choices"][0]["message"]["content"].strip().upper()
            for c in t:
                if c in "ABC": return c
        except Exception:
            time.sleep(2)
    return "C"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--prompts", default="scripts/external_benches/prompts/simpleqa.jsonl")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--num-workers", type=int, default=32)
    a = ap.parse_args()
    gold = {r["prompt"]: r.get("answer", "") for r in map(json.loads, open(a.prompts)) if r.get("prompt")}
    rows = [json.loads(l) for l in open(a.generations) if l.strip()]
    def one(r):
        return grade(r["prompt"], gold.get(r["prompt"], ""), r.get("response", ""))
    with ThreadPoolExecutor(max_workers=a.num_workers) as ex:
        gr = list(ex.map(one, rows))
    n = len(gr); cor = gr.count("A"); inc = gr.count("B"); na = gr.count("C")
    att = cor + inc
    summ = {"run_id": a.run_id, "n": n, "correct": cor, "incorrect": inc, "not_attempted": na,
            "correct_rate": cor / n if n else 0, "incorrect_rate": inc / n if n else 0,
            "not_attempted_rate": na / n if n else 0,
            "accuracy_given_attempted": cor / att if att else 0}
    out = Path("results") / f"{a.run_id}_graded"; out.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(out / "summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2)); print(f"[saved] {out}/summary.json")


if __name__ == "__main__":
    main()
