#!/usr/bin/env python
"""Generative MMLU via vLLM (batched, fast) — the intellectual-suicide probe.

The model must WRITE the answer letter. Log-likelihood MMLU ranks option logprobs
so gibberish still averages ~0.25; here a collapsed model can't emit a valid
letter -> parse_rate ~0, accuracy ~0. parse_rate is the collapse signal.

Runs on any saved HF model dir (incl. a P1b-A checkpoint materialized by
save_p1b_checkpoint.py). Example:
  python experiments/eval_generative_mmlu_vllm.py --model-id outputs/p1b_v7_att_all --n-per-subject 100
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = ["high_school_biology", "college_computer_science",
            "abstract_algebra", "professional_law"]
LABELS = ["A", "B", "C", "D"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True, help="HF model dir or hub id")
    ap.add_argument("--n-per-subject", type=int, default=100)
    ap.add_argument("--subjects", default=",".join(SUBJECTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--out", default=None, help="json summary path")
    args = ap.parse_args()

    import random

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    convs, meta = [], []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        idx = list(range(len(ds)))
        random.Random(args.seed).shuffle(idx)
        for i in idx[: args.n_per_subject]:
            ex = ds[i]
            prompt = f"Question: {ex['question'].strip()}\n"
            for lab, ch in zip(LABELS, ex["choices"]):
                prompt += f"{lab}. {ch}\n"
            prompt += "Answer with a single letter (A, B, C, or D):"
            convs.append([{"role": "user", "content": prompt}])
            meta.append((subject, int(ex["answer"])))

    llm = LLM(model=args.model_id, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=64)
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=8))

    total = len(outs)
    parsed = correct = 0
    for (subject, ans), o in zip(meta, outs):
        text = o.outputs[0].text
        m = re.search(r"[ABCD]", text.upper())
        if m:
            parsed += 1
            correct += int(LABELS.index(m.group()) == ans)
    summary = {
        "model": args.model_id, "n": total,
        "parse_rate": parsed / max(total, 1),          # collapse signal (gibberish -> ~0)
        "accuracy": correct / max(total, 1),           # floor 0, not 0.25
        "acc_of_parsed": correct / max(parsed, 1),     # ~0.25 if coherent-but-wrong
    }
    print(json.dumps(summary, indent=2))
    out = Path(args.out) if args.out else (ROOT / "results" /
          f"gen_mmlu_{Path(args.model_id).name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
