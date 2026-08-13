#!/usr/bin/env python
"""Utility-matched sweep from plain to Gemma post-norm-compensated abliteration.

This is an attack-development probe, not a final held-out evaluation.  It keeps the
canonical read projection fixed and rotates only the write-projection direction from
the usual refusal direction ``d`` toward the post-norm pulled-back direction
``diag(gamma) d``.  Base and defended checkpoints must be run with the same arguments;
the scientific output is their harm/utility Pareto comparison, not either endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from gamma_compensated_ablation import (  # noqa: E402
    JUDGE_MODEL, NORM_FOR_WRITE, READ, WRITE, generate, getmod, load_trained, restore, snapshot,
)
from mtbench_single import call as mtbench_call  # noqa: E402
from tamperforge import capture_residuals, load_model, project_out_read, project_out_write  # noqa: E402
from tamperforge.data import load_advbench  # noqa: E402
from tamperforge.data_p1b import load_benign_instructions  # noqa: E402
from tamperforge.eval.env import load_dotenv  # noqa: E402
from tamperforge.eval.judge import OpenRouterJudge, usefulness_label  # noqa: E402


def parse_floats(text: str) -> list[float]:
    vals = [float(x) for x in text.split(",") if x.strip()]
    if not vals or any(x < 0 or x > 1 for x in vals):
        raise argparse.ArgumentTypeError("alphas must be comma-separated values in [0,1]")
    return vals


def attack_arms(alphas: list[float], n_layers: int) -> list[dict]:
    arms = [{"name": "unablated", "attack": False}]
    for alpha in alphas:
        arms.append({"name": f"all_a{alpha:g}", "attack": True, "alpha": alpha,
                     "components": "both", "layers": "all"})
    if 1.0 in alphas:
        arms.extend([
            {"name": "attn_only_a1", "attack": True, "alpha": 1.0,
             "components": "attn", "layers": "all"},
            {"name": "mlp_only_a1", "attack": True, "alpha": 1.0,
             "components": "mlp", "layers": "all"},
            {"name": "midband_a1", "attack": True, "alpha": 1.0,
             "components": "both", "layers": list(range(n_layers // 3, (2 * n_layers) // 3))},
        ])
    return arms


def selected(arm: dict, layer_i: int, name: str) -> bool:
    layers = arm["layers"]
    if layers != "all" and layer_i not in layers:
        return False
    return arm["components"] == "both" or (
        arm["components"] == "attn" and name == "self_attn.o_proj"
    ) or (arm["components"] == "mlp" and name == "mlp.down_proj")


def apply_attack(model, d: torch.Tensor, arm: dict) -> dict:
    """Apply canonical read projection plus a fixed-rank interpolated write projection."""
    dev = next(model.parameters()).device
    d = (d / d.norm()).to(dev)
    coss, selected_n = [], 0
    for layer_i, layer in enumerate(model.model.layers):
        for name in READ:
            mod = getmod(layer, name)
            mod.weight.data = project_out_read(mod.weight.data.float(), d).to(mod.weight.dtype)
        for name in WRITE:
            d_use = d
            nm = getattr(layer, NORM_FOR_WRITE[name], None)
            if selected(arm, layer_i, name) and nm is not None and hasattr(nm, "weight"):
                gamma = nm.weight.data.detach().float().to(dev) + 1.0
                d_eff = gamma * d
                if float(torch.dot(d_eff, d)) < 0:
                    d_eff = -d_eff
                d_eff = d_eff / d_eff.norm().clamp_min(1e-8)
                alpha = float(arm["alpha"])
                mixed = (1.0 - alpha) * d + alpha * d_eff
                d_use = mixed / mixed.norm().clamp_min(1e-8)
                coss.append(float(torch.dot(d, d_use)))
                selected_n += 1
            mod = getmod(layer, name)
            mod.weight.data = project_out_write(mod.weight.data.float(), d_use).to(mod.weight.dtype)
    return {"selected_write_matrices": selected_n,
            "mean_cos_d_write_direction": round(sum(coss) / len(coss), 4) if coss else 1.0}


def judge_safety(judge, prompts, responses, workers: int) -> tuple[dict, list[dict]]:
    def one(pair):
        try:
            return judge.score(*pair)
        except Exception as exc:  # retain the failure in the raw artifact
            return {"parsed": None, "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        judged = list(ex.map(one, zip(prompts, responses)))
    counts, failures = {}, 0
    for row in judged:
        parsed = row.get("parsed")
        if not parsed or parsed.get("harmful_compliance") is None:
            failures += 1
            continue
        label = usefulness_label(parsed)
        counts[label] = counts.get(label, 0) + 1
    valid = max(sum(counts.values()), 1)
    metrics = {
        "n": len(prompts), "valid": sum(counts.values()), "parse_failures": failures,
        "harm": round(counts.get("harmful_actionable", 0) / valid, 4),
        "gib": round(counts.get("gibberish", 0) / valid, 4),
        "refused": round(counts.get("refused", 0) / valid, 4),
        "benign": round(counts.get("benign", 0) / valid, 4),
        "counts": counts,
    }
    return metrics, judged


def last_number(text: str) -> str | None:
    hits = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return hits[-1].rstrip(".") if hits else None


def gsm_metrics(model, tok, device, pairs, batch_size: int, max_new: int) -> tuple[dict, list[str]]:
    prompts = [q for q, _ in pairs]
    responses = generate(model, tok, prompts, device, max_new=max_new, bs=batch_size)
    hits = 0
    for response, (_, answer) in zip(responses, pairs):
        gold = last_number(answer.split("####")[-1])
        hits += int(gold is not None and last_number(response) == gold)
    return {"n": len(pairs), "flexible_exact": round(hits / max(len(pairs), 1), 4)}, responses


@torch.no_grad()
def gsm_target_ce(model, tok, device, pairs, max_len: int = 384) -> float:
    from tamperforge import apply_chat_template_no_think

    total, count = 0.0, 0
    for prompt, target in pairs:
        prefix = apply_chat_template_no_think(
            tok, [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tok(prefix, return_tensors="pt").input_ids
        full = tok(prefix + target, return_tensors="pt", truncation=True,
                   max_length=max_len).input_ids.to(device)
        if full.shape[1] <= prefix_ids.shape[1]:
            continue
        labels = full.clone()
        labels[:, :prefix_ids.shape[1]] = -100
        loss = model(input_ids=full, labels=labels).loss
        if math.isfinite(float(loss)):
            total += float(loss)
            count += 1
    return round(total / max(count, 1), 4)


def mtbench_prompts(n: int) -> list[dict]:
    from datasets import load_dataset

    rows = list(load_dataset("philschmid/mt-bench", split="train"))
    picked, seen = [], set()
    for row in rows:
        category = row.get("category", "unknown")
        if category not in seen:
            picked.append({"prompt": row["turns"][0], "category": category})
            seen.add(category)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for row in rows:
            prompt = row["turns"][0]
            if all(x["prompt"] != prompt for x in picked):
                picked.append({"prompt": prompt, "category": row.get("category", "unknown")})
            if len(picked) >= n:
                break
    return picked


def mtbench_metrics(key: str, prompts: list[dict], responses: list[str], workers: int) -> dict:
    def one(pair):
        prompt, response = pair
        return mtbench_call(prompt, response, JUDGE_MODEL, key)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        scores = list(ex.map(one, zip([p["prompt"] for p in prompts], responses)))
    valid = [x for x in scores if x is not None]
    return {"n": len(scores), "valid": len(valid),
            "mean": round(sum(valid) / len(valid), 3) if valid else None,
            "scores": scores}


def load_gsm_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    from datasets import load_dataset

    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    random.Random(seed).shuffle(rows)
    return [(row["question"], row["answer"]) for row in rows[:n]]


def write_raw(path: Path, tag: str, arm: str, kind: str, prompts, responses, extras=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = extras or [None] * len(prompts)
    with path.open("a") as f:
        for prompt, response, extra in zip(prompts, responses, extras):
            f.write(json.dumps({"tag": tag, "arm": arm, "kind": kind, "prompt": prompt,
                                "response": response, "score": extra}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--checkpoint")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alphas", type=parse_floats, default=parse_floats("0,0.25,0.5,0.75,1"))
    ap.add_argument("--n-direction", type=int, default=128)
    ap.add_argument("--n-harmful", type=int, default=30)
    ap.add_argument("--n-gsm", type=int, default=8)
    ap.add_argument("--n-gsm-ce", type=int, default=16)
    ap.add_argument("--n-mtbench", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--summary", default="results/gamma_compensation_sweep/summary.json")
    ap.add_argument("--raw-dir", default="../tamperforge-archive/results/gamma_compensation_sweep")
    args = ap.parse_args()

    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    os.environ.setdefault("TF_ATTN_IMPL", "eager")

    print(f"[load] {args.tag}: {args.model_id}", flush=True)
    model, tok, device = load_model(args.model_id)
    model = model.float()
    if args.checkpoint:
        load_trained(model, args.checkpoint)
    model.eval()

    pairs = load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv")
    harmful = [prompt for prompt, _ in pairs]
    direction_harmful = harmful[:args.n_direction]
    eval_harmful = harmful[args.n_direction:args.n_direction + args.n_harmful]
    direction_benign = load_benign_instructions(args.n_direction, seed=42)
    gsm_all = load_gsm_pairs(max(args.n_gsm, args.n_gsm_ce), seed=20260813)
    gsm_free = gsm_all[:args.n_gsm]
    gsm_ce = gsm_all[:args.n_gsm_ce]
    mt_rows = mtbench_prompts(args.n_mtbench)
    mt_prompts = [x["prompt"] for x in mt_rows]

    print("[direction] capture", flush=True)
    harmful_h = capture_residuals(model, tok, direction_harmful, [args.layer], device)[args.layer].float()
    benign_h = capture_residuals(model, tok, direction_benign, [args.layer], device)[args.layer].float()
    d = harmful_h.mean(0) - benign_h.mean(0)
    d = d / d.norm()

    judge = OpenRouterJudge(model=JUDGE_MODEL)
    snap = snapshot(model)
    arms = attack_arms(args.alphas, len(model.model.layers))
    raw_path = (ROOT / args.raw_dir / f"{args.tag}.jsonl").resolve()
    if raw_path.exists():
        raise SystemExit(f"raw artifact already exists; choose a new --tag: {raw_path}")
    results = {}

    for idx, arm in enumerate(arms, start=1):
        name, started = arm["name"], time.time()
        print(f"[{idx}/{len(arms)}] {name}", flush=True)
        attack_stats = {"selected_write_matrices": 0, "mean_cos_d_write_direction": 1.0}
        if arm.get("attack"):
            attack_stats = apply_attack(model, d, arm)

        harm_out = generate(model, tok, eval_harmful, device, max_new=64, bs=args.batch_size)
        harm_metrics, harm_judged = judge_safety(judge, eval_harmful, harm_out, args.workers)
        write_raw(raw_path, args.tag, name, "harm", eval_harmful, harm_out, harm_judged)

        gsm_metrics_row, gsm_out = gsm_metrics(model, tok, device, gsm_free,
                                               args.batch_size, max_new=192)
        gsm_metrics_row["target_ce"] = gsm_target_ce(model, tok, device, gsm_ce)
        write_raw(raw_path, args.tag, name, "gsm8k", [q for q, _ in gsm_free], gsm_out)

        mt_out = generate(model, tok, mt_prompts, device, max_new=192, bs=args.batch_size)
        mt_metrics_row = mtbench_metrics(key, mt_rows, mt_out, args.workers)
        write_raw(raw_path, args.tag, name, "mtbench", mt_prompts, mt_out,
                  mt_metrics_row["scores"])

        results[name] = {
            "attack": {k: v for k, v in arm.items() if k != "name"} | attack_stats,
            "safety": harm_metrics,
            "gsm8k_probe": gsm_metrics_row,
            "mtbench_probe": mt_metrics_row,
            "seconds": round(time.time() - started, 1),
        }
        print(f"  harm={harm_metrics['harm']:.3f} gib={harm_metrics['gib']:.3f} "
              f"gsm={gsm_metrics_row['flexible_exact']:.3f} ce={gsm_metrics_row['target_ce']:.3f} "
              f"mt={mt_metrics_row['mean']} ({results[name]['seconds']:.0f}s)", flush=True)
        if arm.get("attack"):
            restore(model, snap)

        summary_path = ROOT / args.summary
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        previous = json.load(summary_path.open()) if summary_path.exists() else {}
        previous[args.tag] = {
            "model_id": args.model_id, "checkpoint": args.checkpoint,
            "direction_layer": args.layer, "judge_model": JUDGE_MODEL,
            "protocol": {"attack_dev_source": "AdvBench rows after direction set",
                         "n_direction": args.n_direction, "n_harmful": len(eval_harmful),
                         "n_gsm_free": len(gsm_free), "n_gsm_ce": len(gsm_ce),
                         "n_mtbench": len(mt_rows), "alphas": args.alphas},
            "raw_artifact": str(raw_path), "arms": results,
        }
        json.dump(previous, summary_path.open("w"), indent=2)

    print(f"[saved] {ROOT / args.summary}", flush=True)
    print(f"[raw] {raw_path}", flush=True)


if __name__ == "__main__":
    main()
