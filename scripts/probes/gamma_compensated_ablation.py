#!/usr/bin/env python
"""Does gamma-compensating the ablation make Gemma behave like Qwen?

THE HYPOTHESIS UNDER TEST. Weight-space abliteration sets W_out <- W_out - d d^T W_out, so the
block output o has <o, d> = 0 exactly. On Qwen/Llama that lands in the residual stream untouched
(they are pre-norm only). Gemma-3 first passes it through a post-block RMSNorm:

    y = diag(gamma) * (o / rms(o))     =>     <y, d> ∝ <o, diag(gamma) d>

o ⊥ d does NOT imply o ⊥ diag(gamma)d, so the learned gain can rotate the ablated output back
onto the removed direction. Measured (scripts/probes/postnorm_leak_test.py): Gemma's two
post-block norms rotate d with sin ≈ 0.64/0.65 and gamma CV 0.88/0.77 -- the largest of any norm
in either model, and Qwen has no post-block norms at all.

THE FIX THIS TESTS. If the leak is what matters, ablating d is the wrong thing to remove. What
must vanish in the RESIDUAL is <y, d>, i.e. <o, diag(gamma)d>. So project out the PULLED-BACK
direction instead:

    d_eff  =  normalize( diag(gamma) d )      applied to the write matrices feeding that norm

Then <o, diag(gamma)d> = 0 by construction and the direction is actually gone post-norm.
(For pre-norm models gamma is identity from the residual's point of view, d_eff == d, and this
is a no-op -- which is itself a useful control.)

ARMS. For each model: plain rank-1 ablation vs gamma-compensated rank-1 ablation, same
direction, same prompts, judged identically.

  gamma-compensated ~ plain on Gemma  => the leak is not the mechanism; look elsewhere.
  gamma-compensated notably STRONGER  => post-block norms were blunting abliteration, which
                                         means they were also blunting the trained collapse,
                                         and the objective should operate post-norm.

  python scripts/probes/gamma_compensated_ablation.py --model-id google/gemma-3-1b-it \
      --checkpoint <defended.pt> --layer 14 --tag gemma_vg
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
    apply_chat_template_no_think, capture_residuals, is_refusal, load_model,
    project_out_read, project_out_write,
)
from tamperforge.data import load_advbench  # noqa: E402
from tamperforge.data_p1b import load_benign_instructions  # noqa: E402
from tamperforge.eval.judge import OpenRouterJudge, usefulness_label  # noqa: E402

JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731"
READ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj")
WRITE = ("self_attn.o_proj", "mlp.down_proj")
# Which post-block norm rescales each write matrix's output, in Gemma-3.
NORM_FOR_WRITE = {"self_attn.o_proj": "post_attention_layernorm",
                  "mlp.down_proj": "post_feedforward_layernorm"}


def getmod(layer, dotted):
    m = layer
    for p in dotted.split("."):
        m = getattr(m, p)
    return m


def load_trained(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt.pop("_meta", None)
    named = dict(model.named_parameters())
    n = 0
    for name, t in ckpt.items():
        if name in named:
            named[name].data.copy_(t.to(named[name].dtype).to(named[name].device))
            n += 1
    if n == 0:
        sys.exit(f"FATAL: 0 compatible matrices from {ckpt_path}")
    print(f"[ckpt] loaded {n} matrices")


def snapshot(model):
    out = {}
    for i, layer in enumerate(model.model.layers):
        for name in READ + WRITE:
            out[(i, name)] = getmod(layer, name).weight.data.detach().to("cpu", copy=True)
    return out


def restore(model, snap):
    for (i, name), W in snap.items():
        mod = getmod(model.model.layers[i], name)
        mod.weight.data.copy_(W.to(mod.weight.device))


def ablate(model, d, compensate: bool, is_gemma: bool) -> dict:
    """Project d out of every layer. If compensate, use diag(gamma)d on write matrices."""
    dev = next(model.parameters()).device
    d = (d / d.norm()).to(dev)
    stats = {"compensated_layers": 0, "mean_cos_d_deff": 0.0}
    coss = []
    for layer in model.model.layers:
        for name in READ:
            mod = getmod(layer, name)
            W = mod.weight.data.float()
            mod.weight.data = project_out_read(W, d).to(mod.weight.dtype)
        for name in WRITE:
            d_use = d
            if compensate and is_gemma:
                nm = getattr(layer, NORM_FOR_WRITE[name], None)
                if nm is not None and hasattr(nm, "weight"):
                    g = nm.weight.data.detach().float().to(dev) + 1.0   # Gemma stores gain-1
                    d_eff = g * d
                    n = d_eff.norm()
                    if float(n) > 1e-8:
                        d_use = d_eff / n
                        coss.append(float(torch.dot(d_use, d).abs()))
                        stats["compensated_layers"] += 1
            mod = getmod(layer, name)
            W = mod.weight.data.float()
            mod.weight.data = project_out_write(W, d_use).to(mod.weight.dtype)
    if coss:
        stats["mean_cos_d_deff"] = round(sum(coss) / len(coss), 4)
    return stats


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-harmful", type=int, default=30)
    ap.add_argument("--n-direction", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="results/gamma_compensated_ablation/summary.json")
    args = ap.parse_args()

    os.environ.setdefault("TF_ATTN_IMPL", "eager")
    is_gemma = "gemma" in args.model_id.lower()
    print(f"[load] {args.model_id} layer={args.layer} gemma={is_gemma}")
    model, tok, device = load_model(args.model_id)
    model = model.float()
    if args.checkpoint:
        load_trained(model, args.checkpoint)
    model.eval()

    pairs = load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv")
    harmful_all = [p for p, _ in pairs]
    dir_h = harmful_all[:args.n_direction]
    eval_h = harmful_all[args.n_direction:args.n_direction + args.n_harmful]
    benign = load_benign_instructions(args.n_direction, seed=42)

    Hh = capture_residuals(model, tok, dir_h, [args.layer], device)[args.layer].float()
    Hb = capture_residuals(model, tok, benign, [args.layer], device)[args.layer].float()
    d = Hh.mean(0) - Hb.mean(0)
    d = d / d.norm()

    judge = OpenRouterJudge(model=JUDGE_MODEL)
    snap = snapshot(model)
    rows = {}

    outs = generate(model, tok, eval_h, device, args.max_new, args.batch_size)
    rows["unablated"] = judge_all(judge, eval_h, outs)
    print(f"  unablated            harm {rows['unablated']['harm']:.4f} gib {rows['unablated']['gib']:.4f}")

    for comp in (False, True):
        label = "gamma-compensated" if comp else "plain rank-1"
        t0 = time.time()
        st = ablate(model, d, compensate=comp, is_gemma=is_gemma)
        outs = generate(model, tok, eval_h, device, args.max_new, args.batch_size)
        restore(model, snap)
        r = judge_all(judge, eval_h, outs)
        r["kw_refusal"] = round(sum(is_refusal(o) for o in outs) / len(outs), 4)
        r |= st
        rows[label] = r
        extra = (f"  [cos(d,d_eff)={st['mean_cos_d_deff']:.4f} over "
                 f"{st['compensated_layers']} write mats]" if comp and st["compensated_layers"] else "")
        print(f"  {label:20s} harm {r['harm']:.4f} gib {r['gib']:.4f} "
              f"refused {r['refused']:.4f}  ({time.time()-t0:.0f}s){extra}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(p)) if p.exists() else {}
    prev[args.tag] = {"model_id": args.model_id, "checkpoint": args.checkpoint,
                      "layer": args.layer, "n_harmful": len(eval_h), "arms": rows}
    json.dump(prev, open(p, "w"), indent=2)
    print(f"[saved] {p} (tag {args.tag})")


if __name__ == "__main__":
    main()
