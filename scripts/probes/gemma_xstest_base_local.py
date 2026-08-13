#!/usr/bin/env python
"""Gemma BASE over-refusal reference on XSTest-safe. Runs locally on Apple MPS.

WHY. version_J-gemma answers only 41.6% of XSTest safe prompts (refuses 57.6%), which reads as
catastrophic over-refusal. But this project has never measured what Gemma-3-1B-IT does on those
same prompts UNTRAINED, and Gemma is widely reported to be over-refusey out of the box. Without
the base number, 0.416 cannot be attributed: it might be version_J's damage, or it might be
mostly inherited. That single number decides whether version_J is salvageable or disqualified.

The same gap exists for every Gemma arm -- `gbase_clean_xssafe` was never run, so the whole
Gemma campaign has no over-refusal denominator.

DETECTOR. Keyword `is_refusal` by default (free, no API). Pass --judge to additionally score
with the pinned DeepSeek over-refusal judge, which is the instrument that can also see
GIBBERISH -- the failure mode keyword detection is blind to, and the one that invalidated
ABL-v7's Gemma result (61.4% clean gibberish). 250 prompts is roughly $0.05 on the pinned
judge; the repo key had ~$1.67 left when this was written, so this is affordable but check.

  python scripts/probes/gemma_xstest_base_local.py                 # keyword only, free
  python scripts/probes/gemma_xstest_base_local.py --judge         # + judged, ~$0.05
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import apply_chat_template_no_think, is_refusal, load_model  # noqa: E402


def load_xstest_safe() -> list[str]:
    """Prefer the repo's frozen copy; fall back to the Hub."""
    for p in (ROOT / "scripts/external_benches/prompts/xstest_safe.jsonl",
              ROOT / "data/heldout_vg_20260804/xstest_safe.jsonl"):
        if p.exists():
            rows = [json.loads(l) for l in open(p) if l.strip()]
            print(f"[data] {len(rows)} safe prompts from {p.relative_to(ROOT)}")
            return [r["prompt"] for r in rows]
    from datasets import load_dataset
    ds = load_dataset("natolambert/xstest-v2-copy", split="gpt4")
    rows = [r for r in ds if "contrast" not in str(r.get("type", "")).lower()]
    print(f"[data] {len(rows)} safe prompts from the Hub (no frozen copy found)")
    return [r["prompt"] for r in rows]


@torch.no_grad()
def generate(model, tok, prompts, device, max_new, batch_size) -> list[str]:
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        texts = [apply_chat_template_no_think(tok, [{"role": "user", "content": p}],
                                              tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i // batch_size) % 5 == 0:
            print(f"  {min(i+batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--n", type=int, default=-1, help="-1 = all")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--judge", action="store_true", help="also score with the pinned judge (costs API)")
    ap.add_argument("--out", default="results/gemma_base_xstest_safe_local.json")
    args = ap.parse_args()

    os.environ.setdefault("TF_ATTN_IMPL", "eager")
    prompts = load_xstest_safe()
    if args.n > 0:
        prompts = prompts[:args.n]

    print(f"[load] {args.model_id}")
    model, tok, device = load_model(args.model_id)
    if args.dtype == "float32":
        model = model.float()
    model.eval()

    t0 = time.time()
    outs = generate(model, tok, prompts, device, args.max_new, args.batch_size)
    print(f"[gen] {len(outs)} in {time.time()-t0:.0f}s")

    refused = sum(is_refusal(o) for o in outs)
    empty = sum(1 for o in outs if len(o.strip()) < 5)
    res = {"model_id": args.model_id, "n": len(outs),
           "keyword_refusal_rate": round(refused / len(outs), 4),
           "keyword_answered_rate": round(1 - refused / len(outs), 4),
           "near_empty_rate": round(empty / len(outs), 4),
           "detector": "keyword is_refusal; blind to gibberish -- use --judge for that"}
    print(f"\n[keyword] answered {res['keyword_answered_rate']:.4f} | "
          f"refused {res['keyword_refusal_rate']:.4f} | near-empty {res['near_empty_rate']:.4f}")
    print("[compare] version_J-gemma answered 0.416 / refused 0.576 on this same benchmark")

    if args.judge:
        sys.path.insert(0, str(ROOT / "experiments"))
        gen_path = ROOT / "results/gemma_base_xssafe/generations.jsonl"
        gen_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gen_path, "w") as f:
            for p, o in zip(prompts, outs):
                f.write(json.dumps({"prompt": p, "response": o, "condition": "base"}) + "\n")
        print(f"\n[judge] wrote {gen_path.relative_to(ROOT)} -- now run:")
        print(f"  python experiments/judge_generations.py --generations {gen_path.relative_to(ROOT)} "
              f"--run-id gemma_base_xssafe_judged --num-workers 16")
        print("  (kept as a separate step so a judge failure cannot lose the generations)")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res | {"samples": [{"prompt": p, "response": o[:300]}
                                 for p, o in list(zip(prompts, outs))[:5]]},
              open(out, "w"), indent=2)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
