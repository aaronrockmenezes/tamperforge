#!/usr/bin/env python
"""Which bigger model can version_G be tested on, and does it have gemma's post-block norms?

WHY. version_G works on Qwen3-0.6B and degrades to a fortress on Llama-3.2-1B; on gemma-3-1b it
fails, and the two architecture-linked reasons found so far both trace to gemma's POST-BLOCK
RMSNorms: they manufacture a residual that is 96-99.7% a shared DC component (which floored the
uncentred L_rr), and their learned gain shears nearby directions apart (3.2x at cap-rank 16 on
base). Scaling up is only informative if the next model does not reintroduce that confound.

NEITHER NAME NOR COUNT ALONE WORKS, and both traps were hit writing this:

  * Qwen and Llama own a `post_attention_layernorm` that is really their PRE-MLP norm, so
    matching on "post_" hands every model a fake gamma.
  * Qwen3 owns q_norm/k_norm INSIDE self_attn (QK-norm on head dims, off the residual path), so
    a naive module count read Qwen3-0.6B -- the known-good control -- as 4 norms/layer.
  * OLMo-2 owns exactly 2 norms and is NOT pre-norm: it has no input_layernorm at all, and both
    of its norms sit after the sublayer. A count of 2 read it as safe when it is the worst case.

The rule that survives all three: look at the DIRECT CHILDREN of a decoder layer, then

    post-block gain exists  <=>  `post_feedforward_layernorm` present   (gemma-3 style)
                             or  `input_layernorm` absent               (OLMo-2 style)

Models are instantiated on the META device from config alone, so this downloads a few KB of JSON
and no weights.

Also sizes the run. version_G trains --train-scope all, and _reroute_loss holds the ABLATED model
and the FROZEN BASE per step, so the step peak is roughly

    2N (model, bf16) + 2T (W0, trainable only) + 2.5T (overrides + graph) + 2T (grads)
    + 8T (AdamW fp32)  or  2T (adamw8bit)      + activations

for N total and T trainable params.

  python scripts/probes/arch_scout.py
  python scripts/probes/arch_scout.py --models Qwen/Qwen3-4B,mistralai/Mistral-7B-Instruct-v0.3
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM

DEFAULT = [
    "Qwen/Qwen3-0.6B",                   # known-good control: version_G passes all 3 gates here
    "Qwen/Qwen3-1.7B",                   # fits 24GB all-scope fp32, recipe verbatim
    "Qwen/Qwen3-4B-Instruct-2507",       # 48GB w/ adamw8bit, 96GB verbatim
    "Qwen/Qwen3-8B",                     # 96GB w/ adamw8bit
    "Qwen/Qwen3-14B",                    # out of reach on one card
    "HuggingFaceTB/SmolLM3-3B",
    "mistralai/Ministral-8B-Instruct-2410",
    "microsoft/phi-4",
    "ibm-granite/granite-3.3-8b-instruct",
    "allenai/OLMo-2-1124-13B-Instruct",  # POST-BLOCK: no input_layernorm. Disqualified.
    "google/gemma-3-4b-it",              # POST-BLOCK: confirms the confound is family-wide
    "google/gemma-3-1b-it",              # known-bad reference
]

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def probe(model_id: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
    layer0 = layers[0]

    # DIRECT CHILDREN only. Qwen3 owns q_norm/k_norm inside self_attn (QK-norm, applied to head
    # dims), which are not on the residual path -- counting them made Qwen3-0.6B, the known-good
    # pre-norm control, read as 4 norms/layer. Nested norms are never the post-block gain.
    norms = [n for n, m in layer0.named_children() if "norm" in n.lower() and hasattr(m, "weight")]
    post_block = ("post_feedforward_layernorm" in norms
                  or not any("input" in n or n.startswith("pre_") for n in norms))
    # Trainable set = exactly what --train-scope all touches.
    trainable = sum(p.numel() for n, p in layers.named_parameters()
                    if any(k in n for k in ATTN + MLP))
    total = sum(p.numel() for _, p in model.named_parameters())
    return {"id": model_id, "n_layers": len(layers), "norms_per_layer": len(norms),
            "norm_names": norms, "total": total, "trainable": trainable,
            "mlp": mlp_only(layers), "post_block": post_block,
            "hidden": getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size}


def vram(total: int, trainable: int, eightbit: bool) -> float:
    opt = 2 if eightbit else 8
    return (2 * total + (2 + 2.5 + 2 + opt) * trainable) / 2**30 + 1.5   # +acts, rough


def mlp_only(layers) -> int:
    """--train-scope mlp: gate/up/down only. The lever when all-scope will not fit."""
    return sum(p.numel() for n, p in layers.named_parameters() if any(k in n for k in MLP))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT))
    ap.add_argument("--budget-gb", type=float, default=24.0)
    args = ap.parse_args()

    print(f"{'model':40s} {'L':>3s} {'nrm':>3s} {'total':>7s} {'all':>7s} {'8bit':>6s} "
          f"{'mlp8':>6s}  verdict")
    for mid in args.models.split(","):
        try:
            r = probe(mid.strip())
        except Exception as e:                                   # gated / unknown arch
            print(f"{mid.strip():42s}  -- {type(e).__name__}: {str(e)[:44]}")
            continue
        a = vram(r["total"], r["trainable"], False)
        b = vram(r["total"], r["trainable"], True)
        c = vram(r["total"], r["mlp"], True)
        post = r["post_block"]
        if post:
            why = ("gemma-style extra post_feedforward norm"
                   if "post_feedforward_layernorm" in r["norm_names"] else
                   "OLMo-2 style: no input_layernorm, both norms post-block")
            verdict = f"POST-BLOCK -- {why}"
        elif a <= args.budget_gb:
            verdict = "fits, all-scope fp32 -- version_G recipe verbatim"
        elif b <= args.budget_gb:
            verdict = "fits all-scope with --optim adamw8bit"
        elif c <= args.budget_gb:
            verdict = "needs --train-scope mlp + adamw8bit"
        else:
            verdict = "too big for one card"
        print(f"{r['id']:40s} {r['n_layers']:3d} {r['norms_per_layer']:3d} "
              f"{r['total']/1e9:6.2f}B {a:6.1f}G {b:5.1f}G {c:5.1f}G  {verdict}")
    print(f"\nbudget {args.budget_gb:.0f}GB/card. Verdict is the post-block test, not the count: "
          f"see the module docstring for why 2 norms does not mean pre-norm.")


if __name__ == "__main__":
    main()
