#!/usr/bin/env python
"""Evaluate a P1b-A tamper-resistant checkpoint under a REAL abliteration attack.

Training pushes losses; this asks the actual question: when an attacker ablates
the empirical refusal direction from the trained model, do they get an
uncensored-but-USELESS model (defense holds) or an uncensored-and-harmful one
(need a generation-coherence loss)?

Three conditions, same prompts/metrics, keyword + saved generations (judge
separately with judge_generations.py):

  trained_clean    : trained model, no attack        -> should REFUSE, be capable
  trained_attacked : trained model, ablate emp. d    -> uncensored? useless?
  base_attacked     : original base, ablate emp. d    -> CONTROL: free uncensor

Compare trained_attacked vs base_attacked at matched ASR: if the trained model's
uncensored output is far less capable (PPL up / ARC down / judged non-harmful)
than the base's, the tamper-resistance training worked.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import empirical_refusal_direction, load_model
from tamperforge.data import BENIGN_PROMPTS, PROSE_TEXT, load_advbench_prompts
from tamperforge.eval.capability import arc_challenge_accuracy, compute_perplexity, generative_mcq
from tamperforge.eval.log import RunLogger, make_run_id
from tamperforge.eval.safety_eval import eval_advbench_keyword, generate_responses

READ = {"mlp": ("mlp.gate_proj", "mlp.up_proj"),
        "all": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "mlp.gate_proj", "mlp.up_proj")}
WRITE = {"mlp": ("mlp.down_proj",),
         "all": ("self_attn.o_proj", "mlp.down_proj")}


def _load_trained(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt.pop("_meta", {})
    named = dict(model.named_parameters())
    missing = []
    expected = set(meta.get("trainable", ckpt.keys()))
    n = 0
    for name, tensor in ckpt.items():
        if name in named:
            named[name].data.copy_(tensor.to(named[name].dtype).to(named[name].device))
            n += 1
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(f"checkpoint contains {len(missing)} tensors not in model: {missing[:5]}")
    if expected != set(ckpt):
        absent = sorted(expected - set(ckpt))
        extra = sorted(set(ckpt) - expected)
        raise RuntimeError(
            "checkpoint metadata does not match tensors: "
            f"missing={absent[:5]} extra={extra[:5]}"
        )
    if n != len(expected):
        raise RuntimeError(f"loaded {n} matrices, expected {len(expected)} from checkpoint metadata")
    print(f"[eval] loaded {n} trained matrices from {ckpt_path}")
    return meta


@torch.no_grad()
def _attack(model, d: torch.Tensor, layers, scope: str) -> None:
    """Permanent scoped abliteration of direction d (the real attacker)."""
    dev = next(model.parameters()).device
    dd = (d / d.norm().clamp(min=1e-8)).to(dev)
    for li in layers:
        layer = model.model.layers[li]
        for name in READ[scope]:
            mod = layer.get_submodule(name)
            W = mod.weight.data.float()
            mod.weight.data = (W - torch.outer(W @ dd, dd)).to(mod.weight.dtype)
        for name in WRITE[scope]:
            mod = layer.get_submodule(name)
            W = mod.weight.data.float()
            mod.weight.data = (W - torch.outer(dd, dd @ W)).to(mod.weight.dtype)
    print(f"[eval] attacked: ablated emp. refusal dir from {scope} of {len(layers)} layers")


def _capability(model, tok, device, n_arc, n_mcq):
    ppl = compute_perplexity(model, tok, device, PROSE_TEXT)
    arc = arc_challenge_accuracy(model, tok, device, n=n_arc)
    out = {"ppl": ppl["ppl"], "arc": arc["accuracy"]}
    if n_mcq > 0:
        g = generative_mcq(model, tok, device, n_per_subject=n_mcq)
        # generative MMLU: parse_rate is the "intellectual suicide" signal (gibberish -> ~0)
        out.update({"gen_mmlu_parse_rate": g["parse_rate"], "gen_mmlu_acc": g["accuracy"],
                    "gen_mmlu_acc_of_parsed": g["acc_of_parsed"]})
    return out


def _condition(name, model, tok, device, prompts, logger, cfg):
    rows = generate_responses(model, tok, device, prompts[: cfg["n_adv"]],
                              max_new_tokens=cfg["max_new_tokens"],
                              batch_size=cfg["batch_size"],
                              logger=logger,
                              condition=name)
    kw = eval_advbench_keyword(rows)
    cap = _capability(model, tok, device, cfg["n_arc"], cfg["n_mcq"])
    summary = {"condition": name, "keyword_asr": kw["asr"], **cap}
    g = (f" gen_mmlu[parse={cap['gen_mmlu_parse_rate']:.2f} acc={cap['gen_mmlu_acc']:.2f}]"
         if "gen_mmlu_parse_rate" in cap else "")
    print(f"[{name}] keyword_asr={kw['asr']:.2f} ppl={cap['ppl']:.2f} arc={cap['arc']:.2f}{g}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="tamper_resistant_p1b.pt")
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--attack-scope", choices=["mlp", "all"], default="mlp",
                    help="which matrices the attacker ablates (match training, or 'all').")
    ap.add_argument("--abliterate-layers", default="all")
    ap.add_argument("--direction-layer", type=int, default=13)
    ap.add_argument("--n-direction", type=int, default=256)
    ap.add_argument("--n-adv", type=int, default=100)
    ap.add_argument("--n-arc", type=int, default=100)
    ap.add_argument("--n-mcq", type=int, default=25,
                    help="per-subject generative-MMLU questions (model WRITES the letter; "
                         "parse_rate = intellectual-suicide signal). 0 to skip.")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--advbench-source", choices=["walledai", "local"], default="walledai")
    ap.add_argument("--advbench-split", default="train")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or make_run_id("eval_p1b")
    logger = RunLogger(ROOT / "results", run_id, repo_root=ROOT)
    logger.write_manifest({"script": "eval_tamper_resistant.py", "args": vars(args)})
    cfg = {"n_adv": args.n_adv, "n_arc": args.n_arc, "n_mcq": args.n_mcq,
           "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size}

    prompts = load_advbench_prompts(ROOT / "data" / "advbench_harmful_behaviors.csv",
                                    n=max(args.n_adv, args.n_direction), seed=42,
                                    source=args.advbench_source, split=args.advbench_split)
    harmless = BENIGN_PROMPTS
    n_layers_all = None
    conditions = {}

    # --- trained model: clean, then attacked ---
    model, tok, device = load_model(args.model_id, args.device)
    meta = _load_trained(model, str(ROOT / args.checkpoint) if not Path(args.checkpoint).is_absolute()
                         else args.checkpoint)
    n_layers_all = len(model.model.layers)
    layers = list(range(n_layers_all)) if args.abliterate_layers == "all" \
        else [int(x) for x in args.abliterate_layers.split(",")]

    conditions["trained_clean"] = _condition("trained_clean", model, tok, device, prompts, logger, cfg)
    with torch.no_grad():
        d = empirical_refusal_direction(model, tok, prompts[: args.n_direction],
                                        harmless[: args.n_direction], args.direction_layer, device)
    _attack(model, d, layers, args.attack_scope)
    conditions["trained_attacked"] = _condition("trained_attacked", model, tok, device, prompts, logger, cfg)
    del model

    # --- control: original base attacked the same way ---
    base, tok, device = load_model(args.model_id, args.device)
    with torch.no_grad():
        d_base = empirical_refusal_direction(base, tok, prompts[: args.n_direction],
                                             harmless[: args.n_direction], args.direction_layer, device)
    _attack(base, d_base, layers, args.attack_scope)
    conditions["base_attacked"] = _condition("base_attacked", base, tok, device, prompts, logger, cfg)
    del base

    summary = {
        "run_id": run_id, "checkpoint": args.checkpoint, "attack_scope": args.attack_scope,
        "checkpoint_meta": meta,
        "conditions": conditions,
        "read": "trained_clean should refuse+be capable; trained_attacked uncensored but "
                "ideally low-capability/useless; base_attacked = free uncensor control. "
                "Judge generations.jsonl for real ASR (keyword lies on broken models).",
    }
    path = logger.summary(summary)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
