"""Capability and fluency metrics.

The harness intentionally uses transparent local scoring, not hidden benchmark
wrappers. Datasets are loaded through Hugging Face `datasets`; all examples and
aggregate outputs are logged by the calling experiment.
"""

from __future__ import annotations

import math
import random
from typing import Any

import torch
from tqdm.auto import tqdm

from tamperforge.eval.log import RunLogger


def _choice_score(model, tok, device: str, prompt: str, choice: str) -> float:
    """Mean log-prob of *choice* tokens after *prompt*."""
    prompt_ids = tok(prompt, return_tensors="pt").to(device)
    full_ids = tok(prompt + choice, return_tensors="pt").to(device)
    labels = full_ids["input_ids"].clone()
    labels[:, : prompt_ids["input_ids"].shape[1]] = -100
    with torch.no_grad():
        out = model(**full_ids, labels=labels)
    n_toks = (labels != -100).sum().item()
    return -float(out.loss.item()) if n_toks else float("-inf")


def _shuffle_take(ds, n: int | None, seed: int):
    if n is None:
        return ds
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return ds.select(idx[: min(n, len(idx))])


def arc_challenge_accuracy(
    model,
    tok,
    device: str,
    n: int = 100,
    seed: int = 42,
    split: str = "validation",
    logger: RunLogger | None = None,
    condition: str = "condition",
) -> dict[str, Any]:
    """ARC-Challenge multiple-choice log-prob accuracy."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    ds = _shuffle_take(ds, n, seed)
    rows = []
    correct = 0
    total = len(ds)
    if logger:
        logger.event("arc_start", {"condition": condition, "split": split, "n": total})
    for i, ex in tqdm(
        enumerate(ds),
        total=total,
        desc=f"arc:{condition}",
        dynamic_ncols=True,
    ):
        labels = list(ex["choices"]["label"])
        texts = list(ex["choices"]["text"])
        answer = str(ex["answerKey"])
        prompt = "Question: " + ex["question"].strip() + "\nChoices:\n"
        for lab, text in zip(labels, texts):
            prompt += f"{lab}. {text}\n"
        prompt += "Answer:"
        scores = [_choice_score(model, tok, device, prompt, " " + text) for text in texts]
        pred_i = max(range(len(scores)), key=scores.__getitem__)
        pred = labels[pred_i]
        ok = pred == answer
        correct += int(ok)
        if logger:
            logger.event(
                "arc_progress",
                {
                    "condition": condition,
                    "done": i + 1,
                    "total": total,
                    "correct": correct,
                    "accuracy_so_far": correct / max(i + 1, 1),
                },
            )
        rows.append({
            "task": "arc_challenge",
            "i": i,
            "question": ex["question"],
            "choices": dict(zip(labels, texts)),
            "answer": answer,
            "pred": pred,
            "scores": dict(zip(labels, scores)),
            "correct": ok,
        })
    summary = {"task": "arc_challenge", "n": len(rows), "accuracy": correct / max(len(rows), 1), "rows": rows}
    if logger:
        logger.event("arc_done", {"condition": condition, **{k: v for k, v in summary.items() if k != "rows"}})
    return summary


def mmlu_accuracy(model, tok, device: str, subjects: list[str] | None = None,
                  n_per_subject: int = 25, seed: int = 42,
                  split: str = "test",
                  logger: RunLogger | None = None,
                  condition: str = "condition") -> dict[str, Any]:
    """MMLU log-prob accuracy over selected subjects.

    Default subjects keep cost low while covering STEM, humanities, and law.
    """
    if subjects is None:
        subjects = ["high_school_biology", "college_computer_science", "abstract_algebra", "professional_law"]
    from datasets import load_dataset

    rows = []
    correct = 0
    total = 0
    labels = ["A", "B", "C", "D"]
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split=split)
        ds = _shuffle_take(ds, n_per_subject, seed)
        if logger:
            logger.event("mmlu_subject_start", {"condition": condition, "subject": subject, "n": len(ds)})
        for i, ex in tqdm(
            enumerate(ds),
            total=len(ds),
            desc=f"mmlu:{condition}:{subject}",
            dynamic_ncols=True,
        ):
            choices = list(ex["choices"])
            answer_idx = int(ex["answer"])
            prompt = f"Subject: {subject.replace('_', ' ')}\nQuestion: {ex['question'].strip()}\nChoices:\n"
            for lab, text in zip(labels, choices):
                prompt += f"{lab}. {text}\n"
            prompt += "Answer:"
            scores = [_choice_score(model, tok, device, prompt, " " + lab) for lab in labels]
            pred_idx = max(range(len(scores)), key=scores.__getitem__)
            ok = pred_idx == answer_idx
            correct += int(ok)
            total += 1
            if logger:
                logger.event(
                    "mmlu_progress",
                    {
                        "condition": condition,
                        "subject": subject,
                        "done_subject": i + 1,
                        "total_subject": len(ds),
                        "done_all": total,
                        "correct": correct,
                        "accuracy_so_far": correct / max(total, 1),
                    },
                )
            rows.append({
                "task": "mmlu",
                "subject": subject,
                "i": i,
                "question": ex["question"],
                "choices": dict(zip(labels, choices)),
                "answer": labels[answer_idx],
                "pred": labels[pred_idx],
                "scores": dict(zip(labels, scores)),
                "correct": ok,
            })
    summary = {"task": "mmlu", "n": total, "accuracy": correct / max(total, 1), "rows": rows}
    if logger:
        logger.event("mmlu_done", {"condition": condition, **{k: v for k, v in summary.items() if k != "rows"}})
    return summary


def compute_perplexity(
    model,
    tok,
    device: str,
    texts: list[str],
    logger: RunLogger | None = None,
    condition: str = "condition",
) -> dict[str, Any]:
    """Mean token-level perplexity over short prose strings."""
    losses = []
    rows = []
    if logger:
        logger.event("ppl_start", {"condition": condition, "n": len(texts)})
    for i, text in tqdm(
        enumerate(texts),
        total=len(texts),
        desc=f"ppl:{condition}",
        dynamic_ncols=True,
    ):
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        loss = float(out.loss.item())
        losses.append(loss)
        if logger:
            logger.event(
                "ppl_progress",
                {
                    "condition": condition,
                    "done": i + 1,
                    "total": len(texts),
                    "loss": loss,
                    "ppl": math.exp(loss),
                },
            )
        rows.append({"task": "ppl", "i": i, "text": text, "loss": loss, "ppl": math.exp(loss)})
    mean_loss = sum(losses) / max(len(losses), 1)
    summary = {"task": "ppl", "n": len(rows), "loss": mean_loss, "ppl": math.exp(mean_loss), "rows": rows}
    if logger:
        logger.event("ppl_done", {"condition": condition, **{k: v for k, v in summary.items() if k != "rows"}})
    return summary
