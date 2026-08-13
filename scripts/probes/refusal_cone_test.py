#!/usr/bin/env python
"""Is refusal a CONE (many independent directions) or one direction the defence overfit to?

THE OBSERVATION THIS EXPLAINS. On version_G-gemma, surgical k16 reaches 0.94 harm while plain
rank-1 reaches 0.11. The defence stopped the direction it trained against and surgical walked
past it. Two very different mechanisms produce that, and they imply opposite fixes:

  (a) CONE. Refusal is mediated by several mechanistically independent directions
      (arXiv:2502.17420, ICML 2025 -- "concept cones", where orthogonality does NOT imply
      independence under intervention). Deleting the top-1 leaves the rest of the cone intact.
      => harm should RISE STEADILY as you delete rank 1, 2, 4, 8.
      => fix is to train against the cone, not the direction.

  (b) OVERFIT. Refusal is essentially one direction, but the defence memorised that exact
      mean-diff vector, and any perturbed-but-equivalent direction evades it.
      => harm should stay LOW across every rank-k deletion (the SVD directions are all close to
         the mean-diff) yet be HIGH for surgical, whose direction is deliberately rotated away.
      => fix is direction augmentation/jitter during training, a much cheaper change.

The rank-k harm curve separates them cleanly. Run it on a DEFENDED checkpoint -- on a base model
rank-1 already breaks refusal, so there is no signal to read.

GEOMETRY, reported alongside and free. cos(d_meandiff, d_surgical) and the fraction of
d_surgical lying inside the span of the top-k refusal SVD directions. Under (a) surgical should
sit largely INSIDE the refusal subspace (it is another cone direction); under (b) it should sit
largely OUTSIDE (it is the same concept, rotated off the memorised vector).

  python scripts/probes/refusal_cone_test.py --model-id google/gemma-3-1b-it \
      --checkpoint /path/version_g_gemma_500.pt --layer 14
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

from tamperforge import (  # noqa: E402
    abliterate_model_inplace, apply_chat_template_no_think, capture_residuals,
    is_refusal, load_model,
)
from tamperforge.data import load_advbench  # noqa: E402
from tamperforge.data_p1b import load_benign_instructions  # noqa: E402
from tamperforge.eval.judge import OpenRouterJudge, usefulness_label  # noqa: E402

JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731"
TOUCHED = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def load_trained(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt.pop("_meta", None)
    named = dict(model.named_parameters())
    n = 0
    for name, tensor in ckpt.items():
        if name in named:
            named[name].data.copy_(tensor.to(named[name].dtype).to(named[name].device))
            n += 1
    if n == 0:
        sys.exit(f"FATAL: 0 compatible matrices from {ckpt_path} -- wrong architecture?")
    print(f"[ckpt] loaded {n} matrices")


def snapshot(model):
    out = {}
    for i, layer in enumerate(model.model.layers):
        for name in TOUCHED:
            mod = layer
            for part in name.split("."):
                mod = getattr(mod, part)
            out[(i, name)] = mod.weight.data.detach().to("cpu", copy=True)
    return out


def restore(model, snap):
    for (i, name), W in snap.items():
        mod = model.model.layers[i]
        for part in name.split("."):
            mod = getattr(mod, part)
        mod.weight.data.copy_(W.to(mod.weight.device))


@torch.no_grad()
def generate(model, tok, prompts, device, max_new=64, bs=8):
    outs = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        texts = [apply_chat_template_no_think(tok, [{"role": "user", "content": p}],
                                              tokenize=False, add_generation_prompt=True)
                 for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(device)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for j in range(len(chunk)):
            outs.append(tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs


def judge_all(judge, prompts, responses, workers=12):
    from concurrent.futures import ThreadPoolExecutor
    counts, fails = {}, 0

    def one(pr):
        try:
            return judge.score(pr[0], pr[1]).get("parsed")
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for parsed in ex.map(one, zip(prompts, responses)):
            if parsed is None:
                fails += 1
                continue
            lab = usefulness_label(parsed)
            counts[lab] = counts.get(lab, 0) + 1
    n = max(sum(counts.values()), 1)
    return {"harm": round(counts.get("harmful_actionable", 0) / n, 4),
            "gib": round(counts.get("gibberish", 0) / n, 4),
            "refused": round(counts.get("refused", 0) / n, 4), "parse_failures": fails}


@torch.no_grad()
def build_directions(model, tok, device, harmful, benign, cap, layer, max_rank, cap_rank):
    Hh = capture_residuals(model, tok, harmful, [layer], device)[layer].float()
    Hb = capture_residuals(model, tok, benign, [layer], device)[layer].float()
    d1 = Hh.mean(0) - Hb.mean(0)
    d1 = d1 / d1.norm().clamp(min=1e-8)
    D = Hh - Hb.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(D, full_matrices=False)
    R = Vh[:max_rank]                                    # refusal subspace
    Hc = capture_residuals(model, tok, cap, [layer], device)[layer].float()
    Hc = Hc - Hc.mean(0, keepdim=True)
    _, _, Vc = torch.linalg.svd(Hc, full_matrices=False)
    V = Vc[:cap_rank]                                    # capability subspace
    d_surg = d1 - V.T @ (V @ d1)                         # strip capability component
    d_surg = d_surg / d_surg.norm().clamp(min=1e-8)
    return d1, R, d_surg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--checkpoint", default=None, help="defended .pt; omit to test base")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--ranks", default="1,2,4,8")
    ap.add_argument("--cap-rank", type=int, default=16)
    ap.add_argument("--n-harmful", type=int, default=40)
    ap.add_argument("--n-direction", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/refusal_cone_test.json")
    args = ap.parse_args()

    os.environ.setdefault("TF_ATTN_IMPL", "eager")
    print(f"[load] {args.model_id} layer={args.layer} ckpt={args.checkpoint}")
    model, tok, device = load_model(args.model_id)
    model = model.float()
    if args.checkpoint:
        load_trained(model, args.checkpoint)
    model.eval()
    n_layers = len(model.model.layers)

    pairs = load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv")
    harmful_all = [p for p, _ in pairs]
    dir_h = harmful_all[:args.n_direction]
    eval_h = harmful_all[args.n_direction:args.n_direction + args.n_harmful]
    benign = load_benign_instructions(args.n_direction, seed=42)
    cap = load_benign_instructions(args.n_direction * 3, seed=7)[args.n_direction:args.n_direction * 2]

    ranks = [int(x) for x in args.ranks.split(",")]
    d1, R, d_surg = build_directions(model, tok, device, dir_h, benign, cap,
                                     args.layer, max(ranks), args.cap_rank)

    # --- geometry (free, no generation) ---
    cos_d1_surg = float(torch.dot(d1, d_surg).abs())
    geo = {"cos_meandiff_surgical": round(cos_d1_surg, 4)}
    for k in ranks:
        Rk = R[:k]
        inside = float((Rk @ d_surg).norm())        # both unit-ish; ||P_R d_surg||
        geo[f"surgical_in_refusal_span_r{k}"] = round(inside, 4)
        geo[f"meandiff_in_refusal_span_r{k}"] = round(float((Rk @ d1).norm()), 4)
    print("[geometry]", json.dumps(geo))

    judge = OpenRouterJudge(model=JUDGE_MODEL)
    snap = snapshot(model)
    rows = {}

    base_out = generate(model, tok, eval_h, device, args.max_new, args.batch_size)
    rows["unablated"] = judge_all(judge, eval_h, base_out) | {
        "kw_refusal": round(sum(is_refusal(o) for o in base_out) / len(base_out), 4)}
    print(f"  unablated       harm {rows['unablated']['harm']:.4f} gib {rows['unablated']['gib']:.4f}")

    for k in ranks:
        t0 = time.time()
        abliterate_model_inplace(model, R[:k], list(range(n_layers)))
        outs = generate(model, tok, eval_h, device, args.max_new, args.batch_size)
        restore(model, snap)
        rows[f"rank{k}"] = judge_all(judge, eval_h, outs) | {
            "kw_refusal": round(sum(is_refusal(o) for o in outs) / len(outs), 4)}
        print(f"  rank-{k:<2d} SVD      harm {rows[f'rank{k}']['harm']:.4f} "
              f"gib {rows[f'rank{k}']['gib']:.4f}  ({time.time()-t0:.0f}s)")

    abliterate_model_inplace(model, d_surg, list(range(n_layers)))
    outs = generate(model, tok, eval_h, device, args.max_new, args.batch_size)
    restore(model, snap)
    rows[f"surgical_k{args.cap_rank}"] = judge_all(judge, eval_h, outs) | {
        "kw_refusal": round(sum(is_refusal(o) for o in outs) / len(outs), 4)}
    print(f"  surgical k{args.cap_rank:<6d} harm {rows[f'surgical_k{args.cap_rank}']['harm']:.4f} "
          f"gib {rows[f'surgical_k{args.cap_rank}']['gib']:.4f}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(p)) if p.exists() else {}
    prev[args.tag] = {"model_id": args.model_id, "checkpoint": args.checkpoint,
                      "layer": args.layer, "n_harmful": len(eval_h),
                      "geometry": geo, "arms": rows}
    json.dump(prev, open(p, "w"), indent=2)
    print(f"[saved] {p} (tag {args.tag})")
    print("\n[read] harm RISING with rank => CONE (train against the cone). "
          "harm FLAT across ranks but HIGH for surgical => OVERFIT to the mean-diff vector "
          "(direction jitter/augmentation is the cheaper fix).")


if __name__ == "__main__":
    main()
