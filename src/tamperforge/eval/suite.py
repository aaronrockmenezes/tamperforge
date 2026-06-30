"""One-call condition evaluation for P0/P1/P2 experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from tamperforge.data import PROSE_TEXT
from tamperforge.eval.capability import arc_challenge_accuracy, compute_perplexity, mmlu_accuracy
from tamperforge.eval.judge import OpenRouterJudge, judge_responses
from tamperforge.eval.log import RunLogger
from tamperforge.eval.safety_eval import eval_advbench_keyword, generate_responses


@dataclass
class EvalConfig:
    n_advbench: int = 100
    n_arc: int = 100
    n_mmlu_per_subject: int = 25
    seed: int = 42
    max_new_tokens: int = 128
    run_mmlu: bool = False
    run_judge: bool = False
    mmlu_subjects: tuple[str, ...] = (
        "high_school_biology",
        "college_computer_science",
        "abstract_algebra",
        "professional_law",
    )


def evaluate_condition(
    *,
    name: str,
    model,
    tok,
    device: str,
    prompts: list[str],
    config: EvalConfig,
    logger: RunLogger | None = None,
    adapter=None,
    layer: int = 13,
    judge: OpenRouterJudge | None = None,
) -> dict[str, Any]:
    """Run safety, fluency, ARC, optional MMLU, optional judge for one condition."""
    if logger:
        logger.event("condition_start", {"condition": name, "config": asdict(config)})
    responses = generate_responses(
        model,
        tok,
        device,
        prompts[: config.n_advbench],
        adapter=adapter,
        layer=layer,
        max_new_tokens=config.max_new_tokens,
    )
    for row in responses:
        if logger:
            logger.generation({"condition": name, **row})
    keyword = eval_advbench_keyword(responses)
    judge_summary = None
    if config.run_judge:
        if judge is None:
            raise RuntimeError("run_judge=True but no judge provided")
        judge_summary = judge_responses(responses, judge)
        if logger:
            for row in judge_summary["rows"]:
                logger.judgment({"condition": name, **row})
    ppl = compute_perplexity(model, tok, device, PROSE_TEXT)
    arc = arc_challenge_accuracy(model, tok, device, n=config.n_arc, seed=config.seed)
    mmlu = None
    if config.run_mmlu:
        mmlu = mmlu_accuracy(
            model,
            tok,
            device,
            subjects=list(config.mmlu_subjects),
            n_per_subject=config.n_mmlu_per_subject,
            seed=config.seed,
        )
    summary = {
        "condition": name,
        "safety_keyword": {k: v for k, v in keyword.items() if k != "rows"},
        "safety_judge": None if judge_summary is None else {
            k: v for k, v in judge_summary.items() if k != "rows"
        },
        "ppl": {k: v for k, v in ppl.items() if k != "rows"},
        "arc_challenge": {k: v for k, v in arc.items() if k != "rows"},
        "mmlu": None if mmlu is None else {k: v for k, v in mmlu.items() if k != "rows"},
    }
    if logger:
        logger.event("condition_done", summary)
    return {
        "summary": summary,
        "responses": responses,
        "ppl_rows": ppl["rows"],
        "arc_rows": arc["rows"],
        "mmlu_rows": None if mmlu is None else mmlu["rows"],
    }
