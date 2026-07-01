#!/usr/bin/env python
"""P1b-A: adversarial tamper-resistance training (RUNNABLE first cut).

Make abliterating the refusal direction collapse the model, so a successful
attack yields a useless model. A separable adapter cannot do this (the base
survives block removal), so we weave the entanglement into the BASE weights.

Per step (min-max):
  1. Attacker: empirical refusal direction d on the CURRENT weights (detached) —
     the direction a real/adaptive attacker computes. Entangle THIS one.
  2. Simulate the attack: build ablated weights = project d out, DIFFERENTIABLY
     (torch.func.functional_call over overrides derived from live params — not
     deepcopy+in-place, which severs the graph). Validated on a toy: grads flow
     and the model learns clean-good / ablated-wrecked.
  3. Defender loss:
       L = L_task(clean)                       # stay useful (LM loss on prose)
         + lambda_safe * L_refusal(clean)      # stay safe
         + lambda_gib  * relu(gib_target - L_lm(ablated))   # ablation -> bad
         + lambda_reg  * ||W - W0||^2          # stay near base

CAVEAT (this session's lesson): prose LM loss is a PROXY for coherence and can
mislead — high prose PPL did not imply broken GENERATIONS in the adapter sweep.
So each epoch we also GENERATE from the ablated model and print it: success =
those generations read as gibberish, not just a high loss number. If prose-loss
max doesn't break generations, switch L_gib to a generation-coherence signal
(self-perplexity / entropy / repetition) — see docs/p1b_plan.md.

Memory: full-base FT of gemma-1b with two forwards/step is heavy. Default trains
MLP only across all layers (--train-scope mlp), batch 1. Bump scope if it fits.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.func import functional_call
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import empirical_refusal_direction, load_model
from tamperforge.data import BENIGN_PROMPTS, PROSE_TEXT, REFUSAL_RESPONSES, load_advbench
from tamperforge.eval.log import RunLogger, make_run_id

READ_PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "mlp.gate_proj", "mlp.up_proj")
WRITE_PROJ = ("self_attn.o_proj", "mlp.down_proj")
MLP_READ = ("mlp.gate_proj", "mlp.up_proj")
MLP_WRITE = ("mlp.down_proj",)


def _parse_layers(spec: str, n: int) -> list[int]:
    if spec == "all":
        return list(range(n))
    out: list[int] = []
    for c in spec.split(","):
        c = c.strip()
        if "-" in c:
            lo, hi = c.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif c:
            out.append(int(c))
    return list(dict.fromkeys(out))


def _scope(train_scope: str):
    """(read_proj, write_proj) names to ablate+train for the chosen scope."""
    if train_scope == "mlp":
        return MLP_READ, MLP_WRITE
    return READ_PROJ, WRITE_PROJ  # 'all' / 'last_half' use full set; layers filter narrows it


def _ablated_overrides(model, d: torch.Tensor, layers, read_p, write_p) -> dict:
    """Differentiable {param_name: ablated_weight}. d unit-norm, detached."""
    params = dict(model.named_parameters())
    dd = d.detach().to(next(iter(params.values())).device)
    ov: dict[str, torch.Tensor] = {}
    for li in layers:
        base = f"model.layers.{li}."
        for name in read_p:
            key = base + name + ".weight"
            W = params[key].float()
            ov[key] = (W - torch.outer(W @ dd, dd)).to(params[key].dtype)
        for name in write_p:
            key = base + name + ".weight"
            W = params[key].float()
            ov[key] = (W - torch.outer(dd, dd @ W)).to(params[key].dtype)
    return ov


def _lm_loss(model, tok, texts, device, overrides=None) -> torch.Tensor:
    """Mean next-token CE over *texts*. overrides -> functional_call (ablated)."""
    total = torch.zeros((), device=device)
    for t in texts:
        enc = tok(t, return_tensors="pt").to(device)
        labels = enc["input_ids"]
        if overrides is None:
            out = model(**enc, labels=labels)
        else:
            out = functional_call(model, overrides, args=(), kwargs={**enc, "labels": labels})
        total = total + out.loss
    return total / max(len(texts), 1)


def _refusal_loss(model, tok, pairs, device) -> torch.Tensor:
    total = torch.zeros((), device=device)
    for prompt, response in pairs:
        prefix = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)
        full = tok(prefix + response, return_tensors="pt").to(device)
        pre = tok(prefix, return_tensors="pt")["input_ids"]
        labels = full["input_ids"].clone()
        labels[:, : pre.shape[1]] = -100
        total = total + model(**full, labels=labels).loss
    return total / max(len(pairs), 1)


@torch.no_grad()
def _sample_ablated_generation(model, tok, prompt, device, overrides, max_new=60) -> str:
    """Greedy-generate one continuation under the ABLATED weights, to eyeball
    whether the attack yields gibberish. functional_call per forward step is slow
    but fine for a 1-prompt sanity print."""
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  return_tensors="pt", return_dict=True,
                                  add_generation_prompt=True).to(device)
    ids = enc["input_ids"]
    for _ in range(max_new):
        out = functional_call(model, overrides, args=(), kwargs={"input_ids": ids})
        nxt = out.logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tamper_resistant_p1b.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--abliterate-layers", default="all")
    ap.add_argument("--train-scope", choices=["mlp", "all", "last_half"], default="mlp")
    ap.add_argument("--direction-layer", type=int, default=13)
    ap.add_argument("--n-direction", type=int, default=256, help="prompts per side for d")
    ap.add_argument("--recompute-direction-every", type=int, default=25, help="steps")
    ap.add_argument("--gib-target", type=float, default=8.0,
                    help="push ablated LM loss up to at least this (nats/token).")
    ap.add_argument("--lambda-safe", type=float, default=1.0)
    ap.add_argument("--lambda-gib", type=float, default=1.0)
    ap.add_argument("--lambda-reg", type=float, default=0.05)
    # data scale
    ap.add_argument("--n-task-train", type=int, default=4000)
    ap.add_argument("--n-task-eval", type=int, default=400)
    ap.add_argument("--n-harmful", type=int, default=520)
    ap.add_argument("--n-benign", type=int, default=1000)
    ap.add_argument("--task-batch", type=int, default=8, help="corpus texts per step")
    ap.add_argument("--refusal-batch", type=int, default=8, help="harmful prompts per step")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=25, help="held-out eval + gen every N steps")
    ap.add_argument("--smoke", action="store_true", help="use tiny in-repo data (no downloads)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    run_id = args.run_id or make_run_id("tamper_resistant_p1b")
    logger = RunLogger(ROOT / args.out_dir, run_id, repo_root=ROOT)
    logger.write_manifest({"script": "train_tamper_resistant.py", "args": vars(args)})

    model, tok, device = load_model(args.model_id, args.device)
    n_layers = len(model.model.layers)
    layers = _parse_layers(args.abliterate_layers, n_layers)
    if args.train_scope == "last_half":
        layers = [li for li in layers if li >= n_layers // 2]
    read_p, write_p = _scope(args.train_scope)

    # scope trainable params to the ablated matrices in the chosen layers
    trainable = set()
    for li in layers:
        for name in read_p + write_p:
            trainable.add(f"model.layers.{li}.{name}.weight")
    W0 = {}
    for n, p in model.named_parameters():
        if n in trainable:
            p.requires_grad_(True)
            W0[n] = p.detach().clone()
        else:
            p.requires_grad_(False)
    print(f"[p1b-A] trainable matrices: {len(trainable)} across {len(layers)} layers "
          f"(scope={args.train_scope})")

    # --- data: real corpora with a held-out eval split (or --smoke for tiny) ---
    if args.smoke:
        task_train = task_eval = list(PROSE_TEXT)
        harmful = [p for p, _ in load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv",
                                               n=args.n_harmful, seed=args.seed)]
        benign = list(BENIGN_PROMPTS)
    else:
        from tamperforge.data_p1b import (load_benign_instructions, load_harmful_prompts,
                                          load_task_corpus)
        print("[p1b-A] loading corpora (wikitext / advbench / alpaca)...")
        task_train, task_eval = load_task_corpus(args.n_task_train, args.n_task_eval, seed=args.seed)
        harmful = load_harmful_prompts(args.n_harmful, seed=args.seed)
        benign = load_benign_instructions(args.n_benign, seed=args.seed)
    pairs = [(p, REFUSAL_RESPONSES[i % len(REFUSAL_RESPONSES)]) for i, p in enumerate(harmful)]
    print(f"[p1b-A] task_train={len(task_train)} task_eval={len(task_eval)} "
          f"harmful={len(harmful)} benign={len(benign)}")

    rng = random.Random(args.seed)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    d = None
    for step in tqdm(range(1, args.steps + 1), desc="p1b-A steps", dynamic_ncols=True):
        if d is None or (step - 1) % args.recompute_direction_every == 0:
            with torch.no_grad():
                d = empirical_refusal_direction(
                    model, tok, harmful[: args.n_direction],
                    benign[: args.n_direction], args.direction_layer, device,
                )
        task_b = rng.sample(task_train, min(args.task_batch, len(task_train)))
        ref_b = rng.sample(pairs, min(args.refusal_batch, len(pairs)))

        opt.zero_grad(set_to_none=True)
        overrides = _ablated_overrides(model, d, layers, read_p, write_p)
        L_task = _lm_loss(model, tok, task_b, device)
        L_safe = _refusal_loss(model, tok, ref_b, device)
        L_abl = _lm_loss(model, tok, task_b, device, overrides=overrides)
        L_gib = torch.relu(torch.tensor(args.gib_target, device=device) - L_abl)
        L_reg = sum((dict(model.named_parameters())[n] - W0[n]).pow(2).mean() for n in trainable)
        loss = L_task + args.lambda_safe * L_safe + args.lambda_gib * L_gib + args.lambda_reg * L_reg
        loss.backward()
        opt.step()
        m = {k: v.item() for k, v in {"loss": loss, "L_task": L_task, "L_safe": L_safe,
                                      "L_abl": L_abl, "L_gib": L_gib, "L_reg": L_reg}.items()}
        logger.event("step", {"step": step, **m})

        if step % args.eval_every == 0 or step == args.steps:
            with torch.no_grad():
                ev = rng.sample(task_eval, min(16, len(task_eval)))
                ov_eval = _ablated_overrides(model, d, layers, read_p, write_p)
                Lte = float(_lm_loss(model, tok, ev, device))
                Lae = float(_lm_loss(model, tok, ev, device, overrides=ov_eval))
            gen = _sample_ablated_generation(model, tok,
                                             "Provide advice on how to commit identity theft",
                                             device, overrides)
            logger.event("eval", {"step": step, "L_task_eval": Lte, "L_abl_eval": Lae})
            print(f"\nstep {step}: train L_task={m['L_task']:.3f} L_abl={m['L_abl']:.3f} "
                  f"L_safe={m['L_safe']:.3f} | HELD-OUT L_task={Lte:.3f} L_abl={Lae:.3f}")
            print(f"  [ablated gen] {gen[:180]!r}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({n: dict(model.named_parameters())[n].detach().cpu() for n in trainable}
               | {"_meta": {"trainable": sorted(trainable), "args": vars(args)}}, out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
