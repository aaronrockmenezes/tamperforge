#!/usr/bin/env python
"""P1b-A: adversarial tamper-resistance training (SKELETON — not run yet).

Goal that the adapter architecture CANNOT reach: make abliterating the refusal
direction collapse the model into incoherent gibberish, so a successful attack
yields a useless model (smart-and-safe XOR dumb-and-dangerous, in the strong
form). This is impossible with a separable adapter on a coherent base — the base
survives block removal. So P1b-A weaves the entanglement into the BASE weights.

Core idea (min-max, per step):
  1. Attacker: compute the EMPIRICAL refusal direction d on the *current* weights
     (mean harmful - mean harmless). This is what a real/adaptive attacker finds
     — we must entangle THIS direction, not a hand-picked one.
  2. Simulate the attack: ablate d (rank-k projection) from the weights.
  3. Defender: update base weights so that
       - the clean model stays useful (low LM loss on general text) and safe
         (refuses), AND
       - the ABLATED model is incoherent (high perplexity / gibberish).
     i.e. the only way to remove refusal is to also destroy language.

Loss:
  L = L_task(clean)                                   # useful
    + lambda_safe   * L_refusal(clean)               # safe
    + lambda_gib    * L_incoherence(ablated)         # ablation -> GENERATION gibberish
    + lambda_reg    * ||W - W0||^2                    # stay near base (optional)

TARGET GENERATION COHERENCE, NOT PROSE PPL. The P1 ablbase sweep showed prose
perplexity is a MISLEADING proxy: at k=16/32 the entangled model hit PPL 113/809
(teacher-forced on held-out prose) yet its greedy GENERATIONS stayed fluent
English (just vaguer/hedged, which is why judge-ASR fell). Greedy decoding picks
locally high-prob tokens, so high prose-PPL does not imply broken generations.
An attacker who uncensors at k=4 (PPL 23) still gets coherent, actionable harmful
output — a usable model. So the defense objective must make the ABLATED model's
own GENERATIONS incoherent, not just raise a teacher-forced PPL number.

L_incoherence(ablated) should score the ablated model's *self-generated*
continuations for gibberish, e.g. (any/combination):
  - self-perplexity: PPL the ablated model assigns to its OWN greedy generation
    (a coherent model is confident on its own text; a broken one is not), or the
    generation's mean token entropy;
  - degeneracy signals: n-gram repetition rate, unique-token ratio, non-language
    / non-ASCII fraction;
  - a small frozen coherence/fluency classifier or reference-LM PPL over the
    generation.
Verify success on ACTUAL generations reading as gibberish — never on prose PPL
alone (this session's lesson).

The hard/expensive piece is step 2+3: the ablated forward must be DIFFERENTIABLE
in the base weights so lambda_gib pushes them. Use torch.func.functional_call
with weights derived differentiably from the live parameters (do NOT deepcopy +
in-place ablate — that severs the graph). d is detached each step (attacker
recomputes; we don't backprop through direction estimation). NOTE: generation is
non-differentiable (argmax), so L_incoherence needs a differentiable surrogate —
e.g. score coherence via teacher-forcing the ablated model on continuations
sampled/greedy-decoded under stop-grad, or use a soft/Gumbel relaxation. This is
the crux to solve before scaling.

STATUS: skeleton. The marked TODOs are the real work. Do not run until the P1
POC is banked and the differentiable-ablation path is verified on a tiny model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import empirical_refusal_direction, load_model
from tamperforge.data import BENIGN_PROMPTS, PROSE_TEXT, REFUSAL_RESPONSES, load_advbench
from tamperforge.eval.log import RunLogger, make_run_id

# Weight matrices to abliterate + train, per decoder layer (Gemma names).
READ_PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "mlp.gate_proj", "mlp.up_proj")
WRITE_PROJ = ("self_attn.o_proj", "mlp.down_proj")


def _differentiable_ablated_params(model, d: torch.Tensor, layers: list[int]) -> dict:
    """Return a {param_name: tensor} override with d projected out, differentiable.

    d is unit-norm [d_model], detached. For each selected layer:
      READ  W  <- W - (W d) d^T          (input side)
      WRITE W  <- W - d (d^T W)          (output side)
    Built from the live parameters so gradients flow back to them; feed to
    torch.func.functional_call for the ablated forward.

    TODO: assemble the full override dict (only touched params overridden; the
    rest pass through). Handle dtype (compute in fp32). Support rank-k d
    (iterate/orthonormalize) to match the attacker's k.
    """
    dd = d.detach()
    overrides: dict[str, torch.Tensor] = {}
    for li in layers:
        base = f"model.layers.{li}."
        for name in READ_PROJ:
            W = dict(model.named_parameters())[base + name + ".weight"]
            overrides[base + name + ".weight"] = W - torch.outer(W @ dd, dd)
        for name in WRITE_PROJ:
            W = dict(model.named_parameters())[base + name + ".weight"]
            overrides[base + name + ".weight"] = W - torch.outer(dd, dd @ W)
    return overrides


def _perplexity_loss(model, tok, texts, device, param_overrides=None) -> torch.Tensor:
    """Mean LM loss over *texts*. If param_overrides given, run functional_call
    with those weights (the ablated model) so the result is differentiable in the
    live params. TODO: use torch.func.functional_call(model, overrides, args)."""
    raise NotImplementedError("wire torch.func.functional_call for ablated forward")


def _refusal_loss(model, tok, pairs, device) -> torch.Tensor:
    """Teacher-forced LM loss on (harmful prompt -> canned refusal). Same as the
    adapter trainer's refusal loss but on the base weights directly (no adapter).
    TODO: port from train_adapter._refusal_loss, drop the hook."""
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tamper_resistant_p1b.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--abliterate-layers", default="all")
    ap.add_argument("--attack-rank", type=int, default=4,
                    help="k for the simulated rank-k refusal ablation (attacker's minimal k).")
    ap.add_argument("--direction-layer", type=int, default=13)
    ap.add_argument("--n-direction", type=int, default=128)
    ap.add_argument("--ppl-target", type=float, default=1e4,
                    help="C_target: push ablated-model PPL at/above this (gibberish).")
    ap.add_argument("--lambda-safe", type=float, default=1.0)
    ap.add_argument("--lambda-gib", type=float, default=1.0)
    ap.add_argument("--lambda-reg", type=float, default=0.1)
    ap.add_argument("--train-scope", choices=["all", "mlp", "last_half"], default="mlp",
                    help="Which base params get gradients (memory vs coverage).")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    run_id = args.run_id or make_run_id("tamper_resistant_p1b")
    logger = RunLogger(ROOT / args.out_dir, run_id, repo_root=ROOT)
    logger.write_manifest({"script": "train_tamper_resistant.py", "args": vars(args)})

    model, tok, device = load_model(args.model_id, args.device)
    # TODO: freeze/unfreeze by --train-scope; full-base FT is heavy on 1B — start
    #       with mlp-only or last_half to fit the 4090 with grads.
    # TODO: keep a frozen copy W0 for the reg term (or store initial params).

    data = load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv", n=200, seed=args.seed)
    prompts = [p for p, _ in data]
    pairs = [(p, REFUSAL_RESPONSES[i % len(REFUSAL_RESPONSES)]) for i, p in enumerate(prompts)]
    harmless = BENIGN_PROMPTS

    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        # 1. attacker recomputes the empirical refusal direction on CURRENT weights
        with torch.no_grad():
            d = empirical_refusal_direction(
                model, tok, prompts[: args.n_direction],
                harmless[: min(args.n_direction, len(harmless))],
                args.direction_layer, device,
            )  # detached; the attacker's target

        # 2. build differentiable ablated-weight overrides (rank-k d)
        # overrides = _differentiable_ablated_params(model, d, layers)  # TODO layers

        # 3. losses
        # L_task    = _perplexity_loss(model, tok, PROSE_TEXT, device)                 # clean useful
        # L_safe    = _refusal_loss(model, tok, pairs, device)                          # clean safe
        # ppl_abl   = _perplexity_loss(model, tok, PROSE_TEXT, device, overrides)       # ablated
        # L_gib     = torch.relu(args.ppl_target - ppl_abl)   # push ablated -> gibberish
        # L_reg     = param_l2_from_init(model)
        # loss = L_task + args.lambda_safe*L_safe + args.lambda_gib*L_gib + args.lambda_reg*L_reg
        # opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        # logger.event("epoch", {"epoch": epoch, ...})
        raise NotImplementedError(
            "Inner loop not wired. Implement: differentiable ablated forward via "
            "torch.func.functional_call, the three losses, and param scoping. "
            "Verify on a 1-2 layer toy model that d/dW of L_gib is finite and that "
            "L_task and L_gib can co-descend before scaling to gemma-3-1b."
        )

    # torch.save({...}, ROOT / args.out)


if __name__ == "__main__":
    main()
