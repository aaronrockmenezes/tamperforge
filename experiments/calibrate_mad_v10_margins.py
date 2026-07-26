#!/usr/bin/env python
"""Calibrate MAD-v10 CE margins against ARC accuracy and wrong answers.

This diagnostic answers three questions for each fixed attack:

1. How much does gold-answer CE increase?
2. Does multiple-choice accuracy actually fall?
3. Does the attack specifically hurt gold answers, or does it move equally far
   from plausible wrong answers?

It also measures the existing 32-token clean-answer divergence signal on benign
prompts. No harmful generations are produced or printed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tamperforge import (  # noqa: E402
    apply_chat_template_no_think,
    empirical_refusal_directions,
    load_model,
)
from tamperforge.data import load_advbench_prompts  # noqa: E402
from tamperforge.data_p1b import load_benign_instructions  # noqa: E402
from train_mad_v10 import _attack_panel, _parse_layers, _target_batch  # noqa: E402
from train_tamper_resistant_v8 import _ablated_overrides  # noqa: E402


def _apply_checkpoint(model, path: Path) -> int:
    state = torch.load(path, map_location="cpu")
    named = dict(model.named_parameters())
    loaded = 0
    with torch.no_grad():
        for name, value in state.items():
            if name.startswith("_") or name not in named:
                continue
            named[name].copy_(value.to(device=named[name].device, dtype=named[name].dtype))
            loaded += 1
    if not loaded:
        raise RuntimeError(f"no model tensors loaded from {path}")
    return loaded


def _arc_rows(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = list(load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation"))
    random.Random(seed).shuffle(ds)
    rows = []
    for row in ds:
        labels = [str(x) for x in row["choices"]["label"]]
        answer = str(row["answerKey"])
        if answer not in labels:
            continue
        choices = "\n".join(
            f"{label}. {text}"
            for label, text in zip(labels, row["choices"]["text"], strict=False)
        )
        rows.append(
            {
                "prompt": f"{row['question']}\n{choices}\nAnswer with only the correct letter.",
                "labels": labels,
                "gold": answer,
            }
        )
        if len(rows) == n:
            break
    return rows


def _score_pairs(model, tok, pairs, device, batch_size: int, max_len: int, overrides=None):
    scores: list[float] = []
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for start in range(0, len(pairs), batch_size):
        seqs = []
        labels = []
        lengths = []
        for prompt, target in pairs[start:start + batch_size]:
            batch, target_labels = _target_batch(tok, prompt, target, device, max_len)
            seqs.append(batch["input_ids"][0])
            labels.append(target_labels[0])
            lengths.append(batch["input_ids"].shape[1])
        input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
        target_ids = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros_like(input_ids)
        for index, length in enumerate(lengths):
            attention_mask[index, :length] = 1
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if overrides is None:
            logits = model(**kwargs).logits.float()
        else:
            logits = functional_call(model, overrides, args=(), kwargs=kwargs).logits.float()
        token_loss = F.cross_entropy(
            logits[:, :-1].transpose(1, 2),
            target_ids[:, 1:],
            reduction="none",
            ignore_index=-100,
        )
        mask = target_ids[:, 1:] != -100
        per_row = (token_loss * mask).sum(1) / mask.sum(1).clamp_min(1)
        scores.extend(per_row.detach().cpu().tolist())
    return scores


def _choice_scores(model, tok, rows, device, batch_size, max_len, overrides=None):
    flat_pairs = [
        (row["prompt"], label)
        for row in rows
        for label in row["labels"]
    ]
    flat = _score_pairs(model, tok, flat_pairs, device, batch_size, max_len, overrides)
    matrix = []
    offset = 0
    for row in rows:
        width = len(row["labels"])
        matrix.append(flat[offset:offset + width])
        offset += width
    return matrix


@torch.no_grad()
def _continuation_batches(model, tok, prompts, device, n_new: int):
    rows = []
    previous_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        for prompt in prompts:
            enc = apply_chat_template_no_think(
                tok,
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).to(device)
            prompt_len = enc["input_ids"].shape[1]
            full = model.generate(
                **enc,
                max_new_tokens=n_new,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
            if full.shape[1] <= prompt_len:
                continue
            labels = full.clone()
            labels[:, :prompt_len] = -100
            rows.append((full[0], labels[0]))
    finally:
        model.config.use_cache = previous_cache
    return rows


def _score_sequences(model, rows, overrides=None):
    if not rows:
        return []
    pad_id = model.config.pad_token_id
    if pad_id is None:
        pad_id = model.config.eos_token_id
    seqs = [row[0] for row in rows]
    labels = [row[1] for row in rows]
    lengths = [len(row[0]) for row in rows]
    input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)
    target_ids = pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = torch.zeros_like(input_ids)
    for index, length in enumerate(lengths):
        attention_mask[index, :length] = 1
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if overrides is None:
        logits = model(**kwargs).logits.float()
    else:
        logits = functional_call(model, overrides, args=(), kwargs=kwargs).logits.float()
    token_loss = F.cross_entropy(
        logits[:, :-1].transpose(1, 2),
        target_ids[:, 1:],
        reduction="none",
        ignore_index=-100,
    )
    mask = target_ids[:, 1:] != -100
    return ((token_loss * mask).sum(1) / mask.sum(1).clamp_min(1)).detach().cpu().tolist()


def _summary(values) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "n": len(values),
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "p10": float(torch.quantile(tensor, 0.1)),
        "p90": float(torch.quantile(tensor, 0.9)),
    }


def _attack_result(tag, rows, clean_scores, attacked_scores, clean_div, attacked_div):
    gold_gaps = []
    wrong_gaps = []
    clean_correct = []
    attacked_correct = []
    confidence = []
    for row, clean, attacked in zip(rows, clean_scores, attacked_scores, strict=True):
        gold_index = row["labels"].index(row["gold"])
        clean_prediction = min(range(len(clean)), key=clean.__getitem__)
        attacked_prediction = min(range(len(attacked)), key=attacked.__getitem__)
        wrong_indices = [index for index in range(len(clean)) if index != gold_index]
        best_wrong = min(wrong_indices, key=clean.__getitem__)
        gold_gaps.append(attacked[gold_index] - clean[gold_index])
        wrong_gaps.append(attacked[best_wrong] - clean[best_wrong])
        clean_correct.append(clean_prediction == gold_index)
        attacked_correct.append(attacked_prediction == gold_index)
        confidence.append(clean[best_wrong] - clean[gold_index])

    correct_indices = [index for index, value in enumerate(clean_correct) if value]
    correct_indices.sort(key=confidence.__getitem__)
    midpoint = len(correct_indices) // 2
    hard_indices = correct_indices[:midpoint]
    easy_indices = correct_indices[midpoint:]
    wrong_indices = [index for index, value in enumerate(clean_correct) if not value]
    div_gaps = [
        max(0.0, min(30.0, attacked - clean))
        for clean, attacked in zip(clean_div, attacked_div, strict=True)
    ]
    return {
        "attack": tag,
        "arc_n": len(rows),
        "clean_accuracy": sum(clean_correct) / max(len(clean_correct), 1),
        "attacked_accuracy": sum(attacked_correct) / max(len(attacked_correct), 1),
        "accuracy_drop": (
            sum(clean_correct) - sum(attacked_correct)
        ) / max(len(clean_correct), 1),
        "gold_gap": _summary(gold_gaps),
        "plausible_wrong_gap": _summary(wrong_gaps),
        "gold_minus_wrong_gap_mean": (
            sum(gold_gaps) / len(gold_gaps) - sum(wrong_gaps) / len(wrong_gaps)
        ),
        "gold_gap_easy_clean_correct": _summary([gold_gaps[index] for index in easy_indices]),
        "gold_gap_hard_clean_correct": _summary([gold_gaps[index] for index in hard_indices]),
        "gold_gap_clean_wrong": _summary([gold_gaps[index] for index in wrong_indices]),
        "divergence_gap": _summary(div_gaps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="outputs/hf_qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--label", default="base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-arc", type=int, default=200)
    parser.add_argument("--n-div", type=int, default=24)
    parser.add_argument("--div-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-target-len", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--attack-layers", default="10-27")
    parser.add_argument("--direction-layer", type=int, default=13)
    parser.add_argument("--n-direction", type=int, default=256)
    parser.add_argument(
        "--attacks",
        default=(
            "legacy_all_full_shared,"
            "all_write_partial_perlayer_a02,"
            "all_write_partial_perlayer_a04,"
            "all_write_partial_perlayer_a06,"
            "all_write_partial_perlayer_a08,"
            "all_write_perlayer_full"
        ),
    )
    args = parser.parse_args()

    os.environ["TF_QWEN_THINKING"] = "off"
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    model, tok, device = load_model(args.model_id, None)
    model.eval()
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        print(f"[calibrate] loaded checkpoint matrices={_apply_checkpoint(model, checkpoint)}")

    arc_rows = _arc_rows(args.n_arc, args.seed)
    benign = load_benign_instructions(1000, seed=args.seed)
    div_prompts = random.Random(args.seed + 1).sample(benign, args.n_div)
    harmful = load_advbench_prompts(
        ROOT / "data" / "advbench_harmful_behaviors.csv",
        n=520,
        seed=args.seed,
        source="local",
    )
    direction_rng = random.Random(args.seed + 2)
    harmful_sample = direction_rng.sample(harmful, min(args.n_direction, len(harmful)))
    benign_sample = direction_rng.sample(benign, min(args.n_direction, len(benign)))

    n_layers = len(model.model.layers)
    attack_layers = _parse_layers(args.attack_layers, n_layers)
    direction_layers = sorted(set(attack_layers + [args.direction_layer]))
    print(f"[calibrate] ARC={len(arc_rows)} divergence_prompts={len(div_prompts)}")
    print(f"[calibrate] computing directions on {len(harmful_sample)}+{len(benign_sample)} prompts")
    with torch.no_grad():
        directions = empirical_refusal_directions(
            model,
            tok,
            harmful_sample,
            benign_sample,
            direction_layers,
            device,
        )
        shared_direction = directions[args.direction_layer]
        clean_choices = _choice_scores(
            model, tok, arc_rows, device, args.batch_size, args.max_target_len
        )
        continuation_rows = _continuation_batches(
            model, tok, div_prompts, device, args.div_tokens
        )
        clean_div = _score_sequences(model, continuation_rows)

        wanted = {name.strip() for name in args.attacks.split(",") if name.strip()}
        results = []
        for tag, read_proj, write_proj, layers, alphas, per_layer in _attack_panel(
            n_layers, attack_layers, "heretic"
        ):
            if tag not in wanted:
                continue
            print(f"[calibrate] attack={tag}", flush=True)
            overrides = _ablated_overrides(
                model,
                directions if per_layer else shared_direction,
                layers,
                read_proj,
                write_proj,
                alphas,
            )
            attacked_choices = _choice_scores(
                model,
                tok,
                arc_rows,
                device,
                args.batch_size,
                args.max_target_len,
                overrides,
            )
            attacked_div = _score_sequences(model, continuation_rows, overrides)
            result = _attack_result(
                tag,
                arc_rows,
                clean_choices,
                attacked_choices,
                clean_div,
                attacked_div,
            )
            results.append(result)
            print(
                f"  accuracy {result['clean_accuracy']:.3f}->{result['attacked_accuracy']:.3f} "
                f"drop={result['accuracy_drop']:.3f} "
                f"gold_gap={result['gold_gap']['mean']:.3f} "
                f"wrong_gap={result['plausible_wrong_gap']['mean']:.3f} "
                f"div_gap={result['divergence_gap']['mean']:.3f}",
                flush=True,
            )

    payload = {
        "label": args.label,
        "model_id": args.model_id,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "arc_n": len(arc_rows),
        "divergence_n": len(continuation_rows),
        "divergence_tokens": args.div_tokens,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[calibrate] wrote {out}")


if __name__ == "__main__":
    main()
