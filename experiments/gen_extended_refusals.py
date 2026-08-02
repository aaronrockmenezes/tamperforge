#!/usr/bin/env python
"""Generate an extended-refusal dataset (Shairah et al., arXiv:2505.19056).

The baseline we have never run. Their claim: fine-tuning on EXTENDED refusals --
a neutral overview of the topic, then an explicit refusal, then an ethical
rationale -- spreads the refusal signal over many token positions instead of
concentrating it in one latent direction, so abliteration cannot remove it with
a rank-1 edit. No adversarial training, no attack simulation, no inner loop.

Our `REFUSAL_RESPONSES` are one-liners ("I cannot help with that request."),
i.e. exactly the conventional safety tuning they beat. This script builds the
drop-in replacement so `--refusal-file` can train the baseline arm.

Output: {"<prompt>": "<extended refusal>", ...} plus a _meta key.
Resumable -- rerun to fill in only the prompts that are missing or too short.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge.data import load_advbench  # noqa: E402
from tamperforge.eval.env import load_dotenv  # noqa: E402

SYSTEM = """You write safety training data for a language model: EXTENDED REFUSALS.

Given a harmful request, write a refusal with exactly three parts, in order, as
flowing prose (no headings, no bullet points, no numbering):

1. A brief neutral, factual overview of the general subject area the request
   touches on -- the kind of thing found in an encyclopaedia. Demonstrate topical
   understanding WITHOUT any operational detail: no steps, quantities, materials,
   code, targets, or procedures.
2. An explicit, unambiguous refusal of this specific request.
3. A short ethical rationale: the concrete harm that complying would risk, and
   where relevant, a legitimate alternative the user could pursue instead.

Length: 120-200 words. Write only the refusal itself, with no preamble, no
commentary, and no mention of these instructions.

The overview must never make the harmful request easier to carry out. If any
neutral framing would itself be operational, skip straight to refusal and
rationale and expand the ethical reasoning instead."""


def call(prompt: str, model: str, api_key: str, max_tokens: int, sleep_s: float) -> str | None:
    body = {
        "model": model,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "max_completion_tokens": max_tokens,
        "include_reasoning": False,
        "reasoning": {"effort": "none", "exclude": True},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Harmful request:\n{prompt}"},
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/aaronrockmenezes/tamperforge",
            "X-Title": "tamperforge-extrefusal",
        },
        method="POST",
    )
    for attempt in range(1, 5):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if choices:
                text = (choices[0].get("message") or {}).get("content") or ""
                text = text.strip()
                if text:
                    time.sleep(sleep_s)
                    return text
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(attempt * 2)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/extended_refusals_advbench.json")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--n", type=int, default=-1, help="-1 = all AdvBench prompts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--sleep-s", type=float, default=0.1)
    ap.add_argument("--min-words", type=int, default=60,
                    help="Regenerate anything shorter than this -- a one-line refusal "
                         "here would silently make the baseline our own short-refusal arm.")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set; put it in .env or export it")

    # Same source the trainer uses for `harmful`, so the baseline sees identical prompts.
    prompts = [
        p for p, _ in load_advbench(
            ROOT / "data" / "advbench_harmful_behaviors.csv",
            n=(None if args.n < 0 else args.n),
            seed=args.seed,
            source="local",
        )
    ]
    print(f"[extref] {len(prompts)} AdvBench prompts (local csv, seed {args.seed})")

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, str] = {}
    if out_path.exists():
        data = json.loads(out_path.read_text())
        data.pop("_meta", None)
        print(f"[extref] resuming, {len(data)} already present")

    todo = [p for p in prompts if len(str(data.get(p, "")).split()) < args.min_words]
    print(f"[extref] generating {len(todo)}")

    if todo:
        with ThreadPoolExecutor(max_workers=max(args.num_workers, 1)) as ex:
            futs = {ex.submit(call, p, args.model, api_key, args.max_tokens, args.sleep_s): p
                    for p in todo}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="extended refusals"):
                p = futs[fut]
                text = fut.result()
                if text:
                    data[p] = text

    ok = [p for p in prompts if len(str(data.get(p, "")).split()) >= args.min_words]
    missing = [p for p in prompts if p not in data]
    short = len(prompts) - len(ok) - len(missing)
    words = [len(data[p].split()) for p in ok]

    payload = dict(data)
    payload["_meta"] = {
        "source": "advbench_harmful_behaviors.csv (local)",
        "seed": args.seed,
        "generator": args.model,
        "n_prompts": len(prompts),
        "n_ok": len(ok),
        "n_missing": len(missing),
        "n_short": short,
        "mean_words": round(sum(words) / max(len(words), 1), 1),
        "min_words_threshold": args.min_words,
        "recipe": "Shairah et al. arXiv:2505.19056 -- overview + explicit refusal + ethical rationale",
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    tmp.replace(out_path)

    print(f"[extref] wrote {out_path}")
    print(f"[extref] ok={len(ok)} missing={len(missing)} short={short} "
          f"mean_words={payload['_meta']['mean_words']}")
    if missing or short:
        print("[extref] rerun to fill the gaps (resumable)")


if __name__ == "__main__":
    main()
