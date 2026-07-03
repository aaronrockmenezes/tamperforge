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

from tamperforge.data import load_advbench_prompts, load_beavertails, load_harmbench
from tamperforge.eval.env import load_dotenv
from tamperforge.eval.judge import OpenRouterJudge
from tamperforge.eval.log import RunLogger, make_run_id
from tamperforge.eval.safety_eval import eval_advbench_keyword
from tamperforge.eval.suite import EvalConfig, evaluate_condition
from tamperforge.eval.vllm_safety import generate_responses_vllm
from tamperforge.model import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-id", default=None)
    # generic prompt-count for whatever --prompt-source is chosen. -1 (or 0) = FULL
    # dataset (no subset). --n-advbench kept as a deprecated alias for old scripts.
    ap.add_argument("--n-prompts", "--n-advbench", dest="n_prompts", type=int, default=-1,
                    help="number of harmful prompts to eval; -1 or 0 = full dataset")
    ap.add_argument("--n-arc", type=int, default=100)
    ap.add_argument("--n-mmlu-per-subject", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=None,
                    help="Truncate chat-formatted prompt to this many input tokens")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--judge-model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--judge-json-mode", action=argparse.BooleanOptionalAction, default=True,
                    help="Use OpenRouter response_format=json_object when provider supports it")
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--advbench-source", choices=["walledai", "local"], default="walledai")
    ap.add_argument("--advbench-split", default="train")
    ap.add_argument("--advbench-csv", default=str(ROOT / "data" / "advbench_harmful_behaviors.csv"))
    ap.add_argument("--prompt-source", choices=["advbench", "harmbench", "beavertails"], default="advbench",
                    help="harmbench = walledai/HarmBench (gated, needs HF login); "
                         "beavertails = PKU-Alignment/BeaverTails unsafe prompts")
    ap.add_argument("--harmbench-config", choices=["standard", "contextual", "copyright"], default="standard")
    ap.add_argument("--prompt-file", default=None,
                    help="JSONL with a 'prompt' field; overrides --prompt-source. Used for external "
                         "benches (strongreject/jailbreakbench/sorrybench) pre-fetched to a uniform file.")
    ap.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    ap.add_argument("--vllm-batch-size", type=int, default=64)
    ap.add_argument("--vllm-dtype", default="bfloat16")
    ap.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    run_id = args.run_id or make_run_id("p0_baseline")
    logger = RunLogger(ROOT / args.out_dir, run_id, repo_root=ROOT)
    n_prompts = None if args.n_prompts in (-1, 0) else args.n_prompts  # None = full dataset
    cfg = EvalConfig(
        n_advbench=n_prompts if n_prompts is not None else 100000,  # cfg field is int; big = "all"
        n_arc=args.n_arc,
        n_mmlu_per_subject=max(args.n_mmlu_per_subject, 1),
        run_mmlu=args.n_mmlu_per_subject > 0,
        run_judge=args.judge,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )
    manifest = {"script": "p0_baseline_eval.py", "args": vars(args), "eval_config": cfg}
    logger.write_manifest(manifest)

    if args.prompt_file:
        import json as _json
        with open(args.prompt_file) as _f:
            prompts = [_json.loads(l)["prompt"] for l in _f if l.strip()]
        if n_prompts is not None:
            prompts = prompts[:n_prompts]
    elif args.prompt_source == "harmbench":
        prompts = [p for p, _ in load_harmbench(n=n_prompts, seed=cfg.seed, config=args.harmbench_config)]
    elif args.prompt_source == "beavertails":
        prompts = [p for p, _ in load_beavertails(n=n_prompts, seed=cfg.seed)]
    else:
        prompts = load_advbench_prompts(
            args.advbench_csv,
            n=n_prompts,
            seed=cfg.seed,
            source=args.advbench_source,
            split=args.advbench_split,
        )

    if args.backend == "vllm":
        if args.judge:
            raise RuntimeError("For vLLM backend, run judge_generations.py after generation.")
        if args.n_arc > 0 or args.n_mmlu_per_subject > 0:
            raise RuntimeError("For vLLM backend, use lm_eval for capability; set --n-arc 0.")
        rows, device = generate_responses_vllm(
            model_id=args.model_id,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            dtype=args.vllm_dtype,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            batch_size=args.vllm_batch_size,
            logger=logger,
            condition="base",
        )
        keyword = eval_advbench_keyword(rows)
        summary = {
            "run_id": run_id,
            "device": device,
            "backend": "vllm",
            "conditions": {
                "base": {
                    "condition": "base",
                    "safety_keyword": {k: v for k, v in keyword.items() if k != "rows"},
                    "safety_judge": None,
                    "ppl": None,
                    "arc_challenge": None,
                    "mmlu": None,
                }
            },
        }
        path = logger.summary(summary)
        print(json.dumps(summary, indent=2))
        print(f"[saved] {path}")
        return

    model, tok, device = load_model(args.model_id, args.device)
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
