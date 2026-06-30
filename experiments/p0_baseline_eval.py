#!/usr/bin/env python
"""P0 baseline eval: original model, no tamper defense.

Writes:
  results/<run_id>/manifest.json
  results/<run_id>/generations.jsonl
  results/<run_id>/judgments.jsonl          (if --judge)
  results/<run_id>/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge.data import load_advbench_prompts
from tamperforge.eval.env import load_dotenv
from tamperforge.eval.judge import OpenRouterJudge
from tamperforge.eval.log import RunLogger, make_run_id
from tamperforge.eval.suite import EvalConfig, evaluate_condition
from tamperforge.model import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--n-advbench", type=int, default=100)
    ap.add_argument("--n-arc", type=int, default=100)
    ap.add_argument("--n-mmlu-per-subject", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--judge-json-mode", action=argparse.BooleanOptionalAction, default=True,
                    help="Use OpenRouter response_format=json_object when provider supports it")
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    run_id = args.run_id or make_run_id("p0_baseline")
    logger = RunLogger(ROOT / args.out_dir, run_id, repo_root=ROOT)
    cfg = EvalConfig(
        n_advbench=args.n_advbench,
        n_arc=args.n_arc,
        n_mmlu_per_subject=max(args.n_mmlu_per_subject, 1),
        run_mmlu=args.n_mmlu_per_subject > 0,
        run_judge=args.judge,
        max_new_tokens=args.max_new_tokens,
    )
    manifest = {"script": "p0_baseline_eval.py", "args": vars(args), "eval_config": cfg}
    logger.write_manifest(manifest)

    model, tok, device = load_model(args.model_id, args.device)
    prompts = load_advbench_prompts(ROOT / "data" / "advbench_harmful_behaviors.csv",
                                    n=args.n_advbench, seed=cfg.seed)
    judge = OpenRouterJudge(args.judge_model, json_mode=args.judge_json_mode) if args.judge else None
    result = evaluate_condition(
        name="base",
        model=model,
        tok=tok,
        device=device,
        prompts=prompts,
        config=cfg,
        logger=logger,
        judge=judge,
    )
    summary = {"run_id": run_id, "device": device, "conditions": {"base": result["summary"]}}
    path = logger.summary(summary)
    print(json.dumps(summary, indent=2))
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
