#!/usr/bin/env python
"""Build private bad_ul rows from already judged model outputs.

Only rows judged coherent, non-refusal, and harmful-compliant are retained.
The script prints counts and paths only; it never prints prompts or responses.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge.data import REFUSAL_RESPONSES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditions",
        default="trained_attacked,base_attacked",
        help="Comma-separated source conditions eligible for rejected outputs.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    allowed = {item.strip() for item in args.conditions.split(",") if item.strip()}
    best_by_prompt: dict[str, tuple[tuple[int, int], str]] = {}
    read_rows = 0
    eligible_rows = 0

    with source.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            read_rows += 1
            row = json.loads(line)
            if row.get("condition") not in allowed:
                continue
            parsed = ((row.get("judge") or {}).get("parsed") or {})
            if not (
                parsed.get("refusal") == 0
                and parsed.get("harmful_compliance") == 1
                and parsed.get("coherent") == 1
            ):
                continue
            prompt = str(row.get("prompt", "")).strip()
            rejected = str(row.get("response", "")).strip()
            if not prompt or not rejected:
                continue
            eligible_rows += 1
            quality = (
                int(parsed.get("rating") or 0),
                int((row.get("judge") or {}).get("response_chars") or len(rejected)),
            )
            previous = best_by_prompt.get(prompt)
            if previous is None or quality > previous[0]:
                best_by_prompt[prompt] = (quality, rejected)

    prompts = list(best_by_prompt)
    random.Random(args.seed).shuffle(prompts)
    prompts = prompts[: args.limit]
    if len(prompts) < args.limit:
        raise RuntimeError(
            f"only {len(prompts)} distinct eligible prompts; requested {args.limit}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for index, prompt in enumerate(prompts):
            handle.write(
                json.dumps(
                    {
                        "prompt": prompt,
                        "rejected": best_by_prompt[prompt][1],
                        "refusal": REFUSAL_RESPONSES[index % len(REFUSAL_RESPONSES)],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        f"[bad-ul] read={read_rows} eligible={eligible_rows} "
        f"distinct={len(best_by_prompt)} wrote={len(prompts)} out={out}"
    )


if __name__ == "__main__":
    main()
