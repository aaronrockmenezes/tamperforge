#!/usr/bin/env python
"""ABL-v8: adversarial tamper-resistance with a GENERATIVE clean anchor.

v8 changes over v7 (train_tamper_resistant.py):
  - NEW L_clean_gen = lambda_clean * KL(base_gen || clean_gen) on benign prompts.
    v7 anchored clean fidelity only via teacher-forced L_task + weight-reg, both of
    which MISS free-generation drift (same blind spot as MC/ARC/MMLU). Result: the
    ablated->gibberish objective bled into clean generation (IFEval -30/-62/-89% on
    qwen/gemma/llama). L_clean_gen pins the CLEAN model's free-generation distribution
    to the FROZEN BASE on benign inputs, so gibberish stays abliteration-CONDITIONAL.
  - Eval-in-loop: a small self-contained instruction-following probe scored on the
    CLEAN model every --eval-every, logged as clean_ifeval_acc, so clean-capability
    collapse is caught DURING training (not only post-hoc).
Goal: clean IFEval ~= base AND ablated still craters. Emulate Qwen-v7 (near-conditional)
across families. See docs/findings_external_benches_ifeval_2026_07_03.md.

--- original v7 docstring ---
P1b-A: adversarial tamper-resistance training (RUNNABLE first cut).

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
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch.func import functional_call
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import version_a_attack as _VA  # noqa: E402  (--attack-profile version_a only)
import version_c_loop as _VCL  # noqa: E402  (--attack-profile version_c only)

from tamperforge import (apply_chat_template_no_think, empirical_refusal_direction,
                         empirical_refusal_directions, is_refusal, load_model,
                         svd_refusal_directions)
from tamperforge.data import BENIGN_PROMPTS, PROSE_TEXT, REFUSAL_RESPONSES, load_advbench
from tamperforge.eval.judge import OpenRouterJudge, usefulness_label
from tamperforge.eval.log import RunLogger, make_run_id

READ_PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "mlp.gate_proj", "mlp.up_proj")
WRITE_PROJ = ("self_attn.o_proj", "mlp.down_proj")
MLP_READ = ("mlp.gate_proj", "mlp.up_proj")
MLP_WRITE = ("mlp.down_proj",)
ATTN_READ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
ATTN_WRITE = ("self_attn.o_proj",)
SCOPES = {"all": (READ_PROJ, WRITE_PROJ), "mlp": (MLP_READ, MLP_WRITE),
          "attn": (ATTN_READ, ATTN_WRITE)}
HERETIC_WRITE_SCOPES = {
    "all_write": ((), WRITE_PROJ),
    "mlp_write": ((), MLP_WRITE),
    "attn_write": ((), ATTN_WRITE),
}


def _sample_attack(rng, n_layers: int, partial: bool = False, per_layer: bool = False):
    """Random (read_proj, write_proj, layers, alphas, use_per_layer) for ensemble training.

    Tier-1 showed the gibberish-collapse is localized: sub-scope / sub-layer
    ablations under-trigger it (L13-25 leaked at full capability). Randomizing the
    simulated attack each step trains the collapse to fire under ANY of them.

    partial=True adds Heretic's regime: broad layer coverage at partial per-layer
    strength. The full-strength windowed sampler never produces it, which is the
    hole Heretic walked through (docs/heretic_v8_2026_07_18.md, confound check:
    Heretic's edit lands *inside* the protected band at the same magnitude).

    per_layer=True adds the other axis Heretic searches over: each layer loses its
    OWN refusal direction instead of one direction removed everywhere. Sampled, not
    forced — the shared-direction case is the standard Arditi attack the headline
    result is built on, and must stay in the training mix.
    """
    scope = rng.choice(["all", "mlp", "attn"])
    rp, wp = SCOPES[scope]
    kind = rng.choice(["all", "lower", "upper", "rand"] + (["broad"] if partial else []))
    h = n_layers // 2
    if kind == "all":
        layers = list(range(n_layers))
    elif kind == "lower":
        layers = list(range(h))
    elif kind == "upper":
        layers = list(range(h, n_layers))
    elif kind == "broad":  # Heretic shape: nearly all layers, light touch
        layers = list(range(rng.randint(0, h // 2), n_layers))
    else:  # random contiguous window of >= half the layers
        lo = rng.randint(0, h)
        hi = rng.randint(min(lo + h, n_layers - 1), n_layers - 1)
        layers = list(range(lo, hi + 1))
    # ponytail: uniform(0.2,1.0) per layer. Fit the range to a real Heretic sweep's
    # per-layer scales if training overfits one strength band.
    alphas = {li: rng.uniform(0.2, 1.0) for li in layers} if partial and rng.random() < 0.5 else None
    use_pl = per_layer and rng.random() < 0.5
    tag = f"{scope}:{kind}{'' if alphas is None else ':partial'}{':perlayer' if use_pl else ''}"
    return rp, wp, layers, alphas, use_pl, tag


def _sample_attack_profile(
    rng,
    n_layers: int,
    profile: str,
    attack_layers: list[int] | None,
    alpha_min: float,
    alpha_max: float,
    write_scope: str = "mixed_write",
):
    """Sample an interpretable attack profile.

    ``v8`` is the exact full-strength shared-direction ensemble. The other
    profiles isolate Heretic's axes and intentionally modify only output-side
    projections (o_proj/down_proj), matching Heretic rather than adding the
    extra read-projection damage used by the legacy ensemble.
    """
    chosen = profile
    if chosen == "rank1_heretic_mix":
        chosen = rng.choice(["rank1_full", "partial_perlayer"])
    if chosen == "rank1_full":
        return (
            READ_PROJ,
            WRITE_PROJ,
            list(range(n_layers)),
            None,
            False,
            "all:candidate:rank1_full",
        )
    if chosen == "mixed":
        chosen = rng.choice(["v8", "partial_shared", "perlayer_full", "partial_perlayer"])
    if chosen == "v8":
        return _sample_attack(rng, n_layers, partial=False, per_layer=False)
    if not attack_layers:
        raise ValueError(f"attack profile {chosen!r} needs an explicit attack layer band")

    if write_scope == "mixed_write":
        scope = rng.choice(list(HERETIC_WRITE_SCOPES))
    else:
        if write_scope not in HERETIC_WRITE_SCOPES:
            raise ValueError(f"unknown Heretic write scope {write_scope!r}")
        scope = write_scope
    rp, wp = HERETIC_WRITE_SCOPES[scope]
    layers = list(attack_layers)
    use_partial = chosen in {"partial_shared", "partial_perlayer"}
    use_per_layer = chosen in {"perlayer_full", "partial_perlayer"}
    alphas = {li: rng.uniform(alpha_min, alpha_max) for li in layers} if use_partial else None
    tag = f"{scope}:candidate:{chosen}"
    return rp, wp, layers, alphas, use_per_layer, tag


def _attack_metadata(tag, layers, alphas, per_layer, profile) -> dict:
    # alphas is {layer: a} (v8..version_B) or {proj: {layer: a}} (version_C tents) --
    # flatten the latter so the logged alpha stats stay comparable across profiles.
    if alphas and all(isinstance(v, dict) for v in alphas.values()):
        vals = [a for prof in alphas.values() for a in prof.values()]
    else:
        vals = list(alphas.values()) if alphas else []
    return {
        "attack_tag": tag,
        "attack_profile": profile,
        "attack_n_layers": len(layers),
        "attack_layer_min": min(layers) if layers else None,
        "attack_layer_max": max(layers) if layers else None,
        "attack_partial": bool(alphas),
        "attack_per_layer": bool(per_layer),
        "attack_alpha_min": min(vals) if vals else 1.0,
        "attack_alpha_mean": sum(vals) / len(vals) if vals else 1.0,
        "attack_alpha_max": max(vals) if vals else 1.0,
    }


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


def _projection_delta(W: torch.Tensor, d: torch.Tensor, *, write: bool) -> torch.Tensor:
    """Weight delta for removing an orthonormal direction/subspace from a projection."""
    D = d.unsqueeze(0) if d.ndim == 1 else d
    return D.T @ (D @ W) if write else (W @ D.T) @ D


def _ablated_overrides(model, d, layers, read_p, write_p, alphas=None) -> dict:
    """Differentiable {param_name: ablated_weight}. d unit-norm, detached.

    d: one direction/subspace removed from every layer, OR {layer: direction/subspace}
       for the per-layer adaptive attack. A subspace is an orthonormal ``[rank, d]``
       tensor; a direction remains ``[d]`` for backward compatibility.
    alphas: {layer: strength} for partial-strength ablation. None = 1.0 (full
       projection) everywhere, the v8 behaviour.
    """
    params = dict(model.named_parameters())
    dev = next(iter(params.values())).device
    shared = None if isinstance(d, dict) else d.detach().to(dev)
    # alphas is either {layer: strength} (shared across projections, the v8/version_B
    # shape) or {proj_name: {layer: strength}} (independent per-projection tents, which
    # is what Heretic actually searches -- see version_a_attack.Tent).
    per_proj = bool(alphas) and all(isinstance(v, dict) for v in alphas.values())

    def _a(li: int, name: str) -> float:
        if alphas is None:
            return 1.0
        if per_proj:
            return alphas.get(name, {}).get(li, 0.0)
        return alphas[li]

    ov: dict[str, torch.Tensor] = {}
    for li in layers:
        dd = shared if shared is not None else d[li].detach().to(dev)
        base = f"model.layers.{li}."
        for name in read_p:
            a = _a(li, name)
            if a == 0.0:
                continue
            key = base + name + ".weight"
            W = params[key].float()
            ov[key] = (W - a * _projection_delta(W, dd, write=False)).to(params[key].dtype)
        for name in write_p:
            a = _a(li, name)
            if a == 0.0:
                continue
            key = base + name + ".weight"
            W = params[key].float()
            ov[key] = (W - a * _projection_delta(W, dd, write=True)).to(params[key].dtype)
    return ov


def _rownorm_ablated_overrides(model, d, layers, read_p, write_p, alphas=None) -> dict:
    """Differentiable version of heretic's `row_normalization = FULL` ablation.

    heretic/model.py:551-597 does: ablate against ROW-NORMALIZED weights, renormalize the
    rows, restore the original row magnitudes, then compress the delta to a rank-3
    randomized SVD so it fits a LoRA adapter. Everything except the rank-3 truncation is
    differentiable, and the truncation is a storage detail -- the delta carries 0.997 of its
    spectral mass in the top component, so dropping it changes essentially nothing while
    keeping the graph clean. `experiments/version_c_replay.py::heretic_delta` is the exact
    (non-differentiable) form; `tests/test_rownorm_ablation.py` pins them together.

    This is NOT what v8/version_A/version_B trained against -- they used the plain
    `W - a * outer(d, d^T W)`, which differs from heretic's delta by 25-47% relative error.
    """
    params = dict(model.named_parameters())
    dev = next(iter(params.values())).device
    shared = None if isinstance(d, dict) else d.detach().to(dev)
    per_proj = bool(alphas) and all(isinstance(v, dict) for v in alphas.values())

    def _a(li: int, name: str) -> float:
        if alphas is None:
            return 1.0
        if per_proj:
            return alphas.get(name, {}).get(li, 0.0)
        return alphas[li]

    ov: dict[str, torch.Tensor] = {}
    for li in layers:
        dd = shared if shared is not None else d[li].detach().to(dev)
        base = f"model.layers.{li}."
        for name in tuple(read_p) + tuple(write_p):
            a = _a(li, name)
            if a == 0.0:
                continue
            key = base + name + ".weight"
            W = params[key].float()
            rn = W.norm(dim=1, keepdim=True).clamp(min=1e-12)
            Wn = W / rn
            if name in write_p:
                Wa = Wn - a * torch.outer(dd, dd @ Wn)
            else:
                Wa = Wn - a * torch.outer(Wn @ dd, dd)
            Wa = Wa / Wa.norm(dim=1, keepdim=True).clamp(min=1e-12) * rn
            ov[key] = Wa.to(params[key].dtype)
    return ov


_GSM8K_TRAIN_CACHE: list[tuple[str, str]] | None = None


def _load_task_qa(n: int, seed: int = 0) -> list[tuple[str, str]]:
    """(question, full worked answer) pairs from GSM8K TRAIN for the task-degradation loss.

    CONTAMINATION NOTE: lm_eval scores GSM8K *test*, so train/test are disjoint, but this
    still makes GSM8K a less independent capability measure for any run using --gib-mode task.
    Read ARC and MMLU as the uncontaminated check on those runs.
    """
    global _GSM8K_TRAIN_CACHE
    if _GSM8K_TRAIN_CACHE is None:
        from datasets import load_dataset  # noqa: PLC0415
        # canonical id is "openai/gsm8k"; the bare "gsm8k" alias fails HF URI validation
        ds = load_dataset("openai/gsm8k", "main", split="train")
        _GSM8K_TRAIN_CACHE = [(r["question"], r["answer"]) for r in ds]
    rng = random.Random(seed)
    return rng.sample(_GSM8K_TRAIN_CACHE, min(n, len(_GSM8K_TRAIN_CACHE)))


def _gsm8k_answer(text: str) -> str | None:
    """Last number in a completion. GSM8K's own '#### N' marker only appears with few-shot
    priming, which an in-loop probe cannot afford, so score the final number instead. This
    is flexible-extract, NOT the strict-match lm_eval reports -- read it as a trend, and
    never quote it as a GSM8K score."""
    m = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return m[-1].rstrip(".") if m else None


def _clean_gsm8k_probe(model, tok, device, n: int = 8, max_new: int = 256,
                       seed: int = 0) -> float | None:
    """Tiny GSM8K probe on the CLEAN model (eval-in-loop), companion to the ifeval probe.

    Why: TODO.md flags --ifeval-in-loop as a weak clean-capability signal, and 2026-08-02's
    5-prompt smoke test found version_B looping on '2+2=?' while its lm_eval MMLU looked only
    mildly down -- generative degradation that constraint-checking probes miss. Free-generation
    arithmetic catches it live instead of 500 steps later.

    Uses the GSM8K TRAIN split (lm_eval scores test), so this does not contaminate the
    reported number.
    """
    if n <= 0:
        return None
    qa = _load_task_qa(n, seed=seed)
    prev_cache = model.config.use_cache
    model.config.use_cache = True
    hits = 0
    thinking_on = os.environ.get("TF_QWEN_THINKING", "off") == "on"
    try:
        for question, answer in tqdm(qa, desc=f"gsm8k-probe(n={n})", leave=False,
                                     dynamic_ncols=True):
            gold = _gsm8k_answer(answer.split("####")[-1])
            enc = apply_chat_template_no_think(
                tok, [{"role": "user", "content": question}],
                return_tensors="pt", return_dict=True, add_generation_prompt=True,
            ).to(device)
            plen = enc["input_ids"].shape[1]
            gen_kwargs = {"max_new_tokens": max_new, "do_sample": thinking_on,
                          "use_cache": True, "pad_token_id": tok.eos_token_id}
            if thinking_on:
                gen_kwargs.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20})
            with torch.no_grad():
                out = model.generate(**enc, **gen_kwargs)
            got = _gsm8k_answer(tok.decode(out[0, plen:], skip_special_tokens=True))
            hits += int(gold is not None and got is not None and gold == got)
    finally:
        model.config.use_cache = prev_cache
    return hits / max(len(qa), 1)


def _task_degradation_loss(model, tok, qa, device, overrides, max_len: int = 320) -> torch.Tensor:
    """CE of the ABLATED model on KNOWN-CORRECT answers. Want HIGH.

    Why this exists: the argmax-divergence signal it replaces asks the attacked model to pick
    DIFFERENT tokens from the clean model over 32 tokens of benign text. That is satisfiable
    without losing any capability, and measurably was -- version_C reached 0.94 gibberish on
    harmful prompts while holding GSM8K at 0.3715, and gib_ce scored AUC 0.460 (chance) at
    telling attacked from clean. Divergence is not degradation.

    This scores the attacked model against ground-truth answers instead, so the only way to
    raise it is to actually get the task wrong. Loss is masked to the answer tokens; the
    prompt is context, not a target. No generate() call -- one teacher-forced forward per
    pair, so it is cheaper than the 32-step greedy decode it replaces.
    """
    total = torch.zeros((), device=device)
    cnt = 0
    for q, a in qa:
        text = apply_chat_template_no_think(
            tok, [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
        p_ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        full = tok(text + a, return_tensors="pt", truncation=True,
                   max_length=max_len).to(device)
        plen = p_ids["input_ids"].shape[1]
        if full["input_ids"].shape[1] <= plen:
            continue
        labels = full["input_ids"].clone()
        labels[:, :plen] = -100          # score the ANSWER only
        out = (model(**full, labels=labels) if overrides is None
               else functional_call(model, overrides, args=(),
                                    kwargs={**full, "labels": labels}))
        total = total + out.loss
        cnt += 1
    return total / max(cnt, 1)


def _lm_loss(model, tok, texts, device, overrides=None, max_len: int = 256) -> torch.Tensor:
    """Mean next-token CE over *texts*. overrides -> functional_call (ablated)."""
    total = torch.zeros((), device=device)
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        labels = enc["input_ids"]
        if overrides is None:
            out = model(**enc, labels=labels)
        else:
            out = functional_call(model, overrides, args=(), kwargs={**enc, "labels": labels})
        total = total + out.loss
    return total / max(len(texts), 1)


def _refusal_loss(model, tok, pairs, device, overrides=None, max_len: int = 320) -> torch.Tensor:
    """Teacher-forced CE on (harmful prompt -> canned refusal). Low = model
    refuses well. overrides -> functional_call (the ablated model), used to
    check whether ablation REMOVES the refusal (we want it high there).

    max_len caps prefix+target. 320 is ample for the one-line REFUSAL_RESPONSES
    but NOT for extended refusals (--refusal-file, ~190-270 tokens): truncation
    would cut the ethical rationale off the end, which is the very part that
    spreads the refusal signal across token positions. Raise it to match."""
    total = torch.zeros((), device=device)
    for prompt, response in pairs:
        prefix = apply_chat_template_no_think(
            tok,
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 thinking-mode templates leave the assistant inside an open
        # ``<think>`` block. The supervised refusal is the final answer, not hidden
        # reasoning, so close the block before appending the target response.
        target = response
        if os.environ.get("TF_QWEN_THINKING", "off") == "on" and "<think>" in prefix and "</think>" not in prefix:
            target = "</think>\n\n" + response
        full = tok(prefix + target, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        pre = tok(prefix, return_tensors="pt")["input_ids"]
        labels = full["input_ids"].clone()
        labels[:, : pre.shape[1]] = -100
        if overrides is None:
            total = total + model(**full, labels=labels).loss
        else:
            total = total + functional_call(model, overrides, args=(),
                                            kwargs={**full, "labels": labels}).loss
    return total / max(len(pairs), 1)


def _shutdown_loss(model, tok, prompts, device, overrides, target_text: str = "") -> torch.Tensor:
    """CE for a prescribed fail-closed state under attack.

    The default target is immediate EOS. A non-empty target becomes
    ``target_text + EOS``. Unlike gib_ce/harm_ce, this loss creates a specified
    attacked behavior; it does not try to infer coherence from likelihood.
    """
    eos = tok.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else None
    if eos is None:
        raise ValueError("shutdown loss requires tokenizer.eos_token_id")

    total = torch.zeros((), device=device)
    for prompt in prompts:
        prefix = apply_chat_template_no_think(
            tok,
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix_ids = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        if target_text:
            target_ids = tok(
                target_text, return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(device)
        else:
            target_ids = torch.empty((1, 0), dtype=prefix_ids.dtype, device=device)
        eos_ids = torch.tensor([[eos]], dtype=prefix_ids.dtype, device=device)
        target_ids = torch.cat([target_ids, eos_ids], dim=1)
        input_ids = torch.cat([prefix_ids, target_ids], dim=1)
        labels = input_ids.clone()
        labels[:, : prefix_ids.shape[1]] = -100
        out = functional_call(
            model,
            overrides,
            args=(),
            kwargs={
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            },
        )
        total = total + out.loss
    return total / max(len(prompts), 1)


def _argmax_divergence_loss(model, tok, prompts, device, overrides, n_new: int = 32) -> torch.Tensor:
    """Generation-targeted gib signal (replaces prose-PPL).

    For each prompt: greedily generate the CLEAN model's continuation (no grad),
    then teacher-force the ABLATED model on (prompt + clean continuation) and take
    the CE on the continuation tokens. High CE = the ablated model can't reproduce
    the coherent text the clean model generates -> ablation destroys GENERATION
    ability, not just teacher-forced prose perplexity. Returns mean CE (want HIGH).

    Prose-PPL failed because greedy decoding stays fluent at high teacher-forced
    PPL; scoring the ablated model against the clean model's OWN greedy tokens
    attacks the argmax directly.
    """
    total = torch.zeros((), device=device)
    cnt = 0
    for prompt in prompts:
        enc = apply_chat_template_no_think(
            tok,
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).to(device)
        plen = enc["input_ids"].shape[1]
        prev_cache = model.config.use_cache
        model.config.use_cache = True
        with torch.no_grad():
            full = model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                                  use_cache=True, pad_token_id=tok.eos_token_id)
        model.config.use_cache = prev_cache
        if full.shape[1] <= plen:
            continue
        labels = full.clone()
        labels[:, :plen] = -100
        # CE in fp32 with a per-token clamp. The ablated model can assign ~0 prob to a
        # token (log(0)) which overflows bf16 to inf/1e30 -> spikes gib_ce -> non-finite
        # loss -> step skipped/poisoned. That's the "seed lottery" (only stable seeds
        # trained). fp32 logsumexp + clamp(max=30) >> gap_target keeps gib_ce bounded and
        # deterministic without changing the objective region.
        logits = functional_call(model, overrides, args=(),
                                 kwargs={"input_ids": full}).logits.float()
        sl = logits[:, :-1, :].reshape(-1, logits.size(-1))
        tl = labels[:, 1:].reshape(-1)
        ce_tok = torch.nn.functional.cross_entropy(sl, tl, ignore_index=-100, reduction="none")
        mask = tl != -100
        ce = torch.clamp(ce_tok[mask], max=30.0).mean() if mask.any() else torch.zeros((), device=device)
        total = total + ce
        cnt += 1
    return total / max(cnt, 1)


def _reroute_loss(model, tok, pairs, device, W0, overrides, layers, max_len: int = 320,
                  center: bool = False):
    """v9 option E: Circuit-Breakers representation rerouting, applied under ATTACK.

    For each (harmful prompt, real harmful completion): teacher-force both the ABLATED
    model (overrides) and the FROZEN BASE (W0) on the same text, and penalise positive
    cosine similarity between their hidden states over *layers*. Want LOW: after the
    attack the model must not process harmful content the way the base does.

    Why this and not another cross-entropy: every reference-text CE tried (gib_ce,
    harm_ce on the affirmative prefix, harm_ce on mined completions) is blind where it
    matters, because likelihood cannot measure coherence — degenerate repetition is the
    most likely text there is. This is a representation-space target, so it has no
    reference string to saturate against. relu() so only positive similarity is pushed
    (orthogonal is enough; anti-aligned is not rewarded further) — Zou et al. 2024,
    arXiv:2406.04313.

    NOTE: this does NOT replace gib_ce. gib_ce does two jobs — the gibberish wall AND
    the benign capability crater (ARC -25%, MMLU -32%, GSM8K -95%) that is the MAD
    headline. Rerouting only touches harmful-input processing and says nothing about
    GSM8K. E supplements, never substitutes.

    `center` subtracts the per-position mean over the sequence from BOTH streams before the
    cosine. Default False so every existing run reproduces bit-identically; pass it for gemma.

    WHY IT MATTERS (measured 2026-08-13, results/rr_gradient_scale.json + DC probe). A residual
    stream is dominated by a shared DC component that carries no content, and gemma's is extreme:
    ||mean|| / RMS||h|| is 0.997 at layer 13 and stays above 0.958 through layer 25, versus
    0.85-0.96 for Qwen. Content is ~0.3% of gemma's hidden state by norm. Since h_att and h_ref
    both carry that same component, an uncentred cosine cannot fall below roughly
    (||mu||/||h||)^2 unless the DC component itself moves -- a floor of ~0.96 on gemma against
    ~0.81 on Qwen.

    That is not a theoretical concern, it is what actually happened: gemma version_G's L_rr went
    0.9866 -> 0.9522 in 500 steps and stopped, i.e. it walked exactly to its DC floor and parked
    there, while Qwen went 0.9854 -> 0.2458, straight through its own. Pushing lower on gemma
    requires moving a component that is 99.7% of the residual and load-bearing for everything the
    model does -- which lambda_clean and lambda_reg exist to forbid. So the rerouting objective
    and the clean-preservation objectives are in direct conflict on gemma and barely interact on
    Qwen, and L_rr has almost no dynamic range to work in.

    Centred, the two architectures look the same: cross-prompt cosine drops to ~-0.18 for BOTH,
    which is just the -1/(n-1) of decorrelated vectors. So all of the excess similarity was DC,
    and removing it restores the metric's range.

    This is NOT the same as the earlier gradient-starvation guess, which was measured and
    falsified: gemma's ||dL_rr||/||dL_lm|| is only 2.1x below Qwen's, nowhere near enough to
    explain a run that never moved.
    """
    total = torch.zeros((), device=device)
    cnt = 0
    for prompt, completion in pairs:
        prefix = apply_chat_template_no_think(
            tok, [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tok(prefix + completion, return_tensors="pt",
                  truncation=True, max_length=max_len).to(device)
        plen = min(tok(prefix, return_tensors="pt")["input_ids"].shape[1],
                   enc["input_ids"].shape[1] - 1)
        h_att = functional_call(model, overrides, args=(),
                                kwargs={**enc, "output_hidden_states": True}).hidden_states
        with torch.no_grad():
            h_ref = functional_call(model, W0, args=(),
                                    kwargs={**enc, "output_hidden_states": True}).hidden_states
        for li in layers:
            a = h_att[li][:, plen:, :].float()
            b = h_ref[li][:, plen:, :].float()
            if a.shape[1] == 0:
                continue
            if center:
                # Estimate the DC component from the FROZEN BASE only, and subtract the same
                # vector from both streams. Using each stream's own mean would let the attacked
                # model shrink the loss by shifting its mean rather than by rerouting content,
                # which is the degenerate solution this whole term exists to avoid.
                mu = b.mean(dim=1, keepdim=True).detach()
                a, b = a - mu, b - mu
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            total = total + torch.relu(cos).mean()
            cnt += 1
    return total / max(cnt, 1)


def _clean_gen_kl_loss(
    model, tok, prompts, device, W0, n_new: int = 32, overrides=None
) -> torch.Tensor:
    """GENERATIVE clean-fidelity anchor (v8). For each benign prompt: greedily
    generate the CLEAN (live) model's continuation, then teacher-force both the CLEAN
    (trainable) and the FROZEN BASE (W0 via functional_call) models on it, and return
    KL(base || clean) on the continuation tokens. Want LOW: the clean model must keep
    the base model's benign next-token distribution under free generation. Counters the
    generative drift that teacher-forced L_task cannot see (the -30/-62/-89% IFEval tax).
    """
    total = torch.zeros((), device=device)
    cnt = 0
    for prompt in prompts:
        enc = apply_chat_template_no_think(
            tok, [{"role": "user", "content": prompt}],
            return_tensors="pt", return_dict=True, add_generation_prompt=True,
        ).to(device)
        plen = enc["input_ids"].shape[1]
        prev_cache = model.config.use_cache
        model.config.use_cache = True
        with torch.no_grad():
            full = model.generate(**enc, max_new_tokens=n_new, do_sample=False,
                                  use_cache=True, pad_token_id=tok.eos_token_id)
        model.config.use_cache = prev_cache
        if full.shape[1] <= plen:
            continue
        clean_logits = (
            model(input_ids=full).logits.float()
            if overrides is None
            else functional_call(
                model, overrides, args=(), kwargs={"input_ids": full}
            ).logits.float()
        )
        with torch.no_grad():
            base_logits = functional_call(model, W0, args=(),
                                          kwargs={"input_ids": full}).logits.float()
        # continuation positions predict tokens plen..end (logits at plen-1..-2)
        sl_clean = clean_logits[:, plen - 1:-1, :]
        sl_base = base_logits[:, plen - 1:-1, :]
        logp_clean = torch.log_softmax(sl_clean, dim=-1)
        logp_base = torch.log_softmax(sl_base, dim=-1)
        p_base = logp_base.exp()
        kl = (p_base * (logp_base - logp_clean)).sum(-1)             # KL(base||clean) per pos
        kl = torch.clamp(kl, max=30.0).mean()
        total = total + kl
        cnt += 1
    return total / max(cnt, 1)


_IFEVAL_PROBE = [
    ("Write a sentence about the ocean that contains the word 'lantern'.", lambda t: "lantern" in t.lower()),
    ("Reply with exactly three words.", lambda t: len(t.split()) == 3),
    ("List three fruits as a numbered list 1. 2. 3.", lambda t: all(f"{i}." in t for i in (1, 2, 3))),
    ("Answer in all capital letters: what colour is the sky on a clear day?", lambda t: any(c.isalpha() for c in t) and t.upper() == t),
    ("Write one sentence that does NOT contain the letter e.", lambda t: "e" not in t.lower() and len(t.strip()) > 3),
    ("Respond with a single word.", lambda t: len(t.split()) == 1),
    ("Write two sentences, each ending with an exclamation mark.", lambda t: t.count("!") >= 2),
    ("Give a greeting that starts with the word 'Hello'.", lambda t: t.strip().lower().startswith("hello")),
    ("Write a sentence that includes the word 'banana'.", lambda t: "banana" in t.lower()),
    ("Respond using only lowercase letters.", lambda t: any(c.isalpha() for c in t) and t.lower() == t),
    ("Include the number 42 somewhere in your reply.", lambda t: "42" in t),
    ("End your response with the word 'done'.", lambda t: t.strip().lower().rstrip(".!").endswith("done")),
    ("Write a sentence that contains a comma.", lambda t: "," in t),
    ("Reply with exactly four words.", lambda t: len(t.split()) == 4),
    ("Write a short reply containing no digits.", lambda t: len(t.strip()) > 0 and not any(c.isdigit() for c in t)),
    ("Start your response with the word 'The'.", lambda t: t.strip().startswith("The")),
    ("Write a sentence containing both 'ocean' and 'blue'.", lambda t: "ocean" in t.lower() and "blue" in t.lower()),
    ("Respond with exactly five words.", lambda t: len(t.split()) == 5),
    ("Include the word 'because' in your answer.", lambda t: "because" in t.lower()),
    ("Include a colon (:) in your response.", lambda t: ":" in t),
    ("End your response with a question mark.", lambda t: t.strip().endswith("?")),
    ("Write a sentence containing the word 'sunset'.", lambda t: "sunset" in t.lower()),
    ("Respond with exactly two words.", lambda t: len(t.split()) == 2),
    ("Write a reply that contains the word 'quantum'.", lambda t: "quantum" in t.lower()),
]


_ARC_CACHE: list | None = None
_MMLU_CACHE: list | None = None
# same 12 subjects as scripts/v11_cap_eval.sh, so in-loop numbers are comparable to the
# campaign's lm_eval MMLU figures rather than being a different slice of the benchmark.
MMLU_SUBJECTS = ("high_school_biology", "college_computer_science", "abstract_algebra",
                 "machine_learning", "philosophy", "world_religions",
                 "high_school_us_history", "econometrics", "sociology",
                 "professional_medicine", "business_ethics", "computer_security")


def _mmlu_rows(n: int, seed: int = 1234) -> list[tuple[str, list[str], int]]:
    """MMLU as (question, choice_texts, answer_index) over the campaign's 12 subjects."""
    global _MMLU_CACHE
    if _MMLU_CACHE is None:
        from datasets import load_dataset
        rows = []
        for subj in MMLU_SUBJECTS:
            try:
                ds = load_dataset("cais/mmlu", subj, split="test")
            except Exception as e:  # noqa: BLE001
                print(f"[mmlu-probe] {subj} unavailable ({type(e).__name__})")
                continue
            head = ("The following are multiple choice questions (with answers) about "
                    f"{subj.replace('_', ' ')}.\n\n")
            for r in ds:
                ch = [str(c) for c in r["choices"]]
                # lm_eval mmlu scores the LETTER after a subject-headed lettered list, not
                # the answer text. Scoring the text instead reads ~0.30 where the real
                # number is ~0.44 -- a silent 4.8-sigma error.
                body = "\n".join(f"{L}. {c}" for L, c in zip("ABCD", ch, strict=False))
                rows.append((f"{head}{r['question']}\n{body}\nAnswer:",
                             [f" {L}" for L in "ABCD"[: len(ch)]], int(r["answer"])))
        random.Random(seed).shuffle(rows)
        _MMLU_CACHE = rows
    return _MMLU_CACHE[:n]


def _arc_rows(n: int, seed: int = 1234) -> list[tuple[str, list[str], int]]:
    """ARC-Challenge as (question, choice_texts, answer_index). Cached across calls."""
    global _ARC_CACHE
    if _ARC_CACHE is None:
        from datasets import load_dataset
        # lm_eval's arc_challenge scores the TEST split; using validation here would make
        # in-loop numbers quietly incomparable to every campaign ARC figure.
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        rows = []
        for r in ds:
            labels = list(r["choices"]["label"])
            texts = list(r["choices"]["text"])
            if str(r["answerKey"]) not in labels:
                continue
            # lm_eval arc_challenge: context "Question: ...\nAnswer:", continuation is the
            # answer TEXT.
            rows.append((f"Question: {r['question']}\nAnswer:",
                         [f" {x}" for x in texts], labels.index(str(r["answerKey"]))))
        random.Random(seed).shuffle(rows)
        _ARC_CACHE = rows
    return _ARC_CACHE[:n]


@torch.no_grad()
def _clean_mc_probe(model, tok, device, rows, batch_size: int = 8) -> float:
    """ARC-Challenge accuracy by LOGLIKELIHOOD -- no generation.

    Why this exists alongside the IFEval probe: IFEval is rule-based constraint following
    ("use exactly three headers", "answer with a bulleted list"). At the in-loop budget of
    48 new tokens most of those constraints cannot physically be satisfied, so a large part
    of the score is measuring truncation rather than capability.

    Scoring each choice as a continuation needs only forward passes, so this is also the
    CHEAPER probe: n=100 is 400 batched forwards, against roughly 1150 sequential decode
    steps for a 24-prompt IFEval probe. Decode is ~68% of step time on this model.

    Format matches lm_eval's arc_challenge (`Question: ...\\nAnswer:` + " {choice}", summed
    logprob, unnormalised `acc`) so in-loop numbers are comparable to the lm_eval runs the
    campaign reports.
    """
    if not rows:
        return float("nan")
    flat = [(ctx, cont, qi)
            for qi, (ctx, conts, _) in enumerate(rows) for cont in conts]
    scores: dict[int, list[float]] = {}
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    old_side, tok.padding_side = tok.padding_side, "right"
    try:
        for i in range(0, len(flat), batch_size):
            chunk = flat[i: i + batch_size]
            plens = [len(tok(p, add_special_tokens=True)["input_ids"]) for p, _, _ in chunk]
            enc = tok([p + c for p, c, _ in chunk], return_tensors="pt",
                      padding=True, truncation=True, max_length=512).to(device)
            ids = enc["input_ids"]
            # NOT .float(): [B, L, 152k] in fp32 is ~10GB at batch 32 and OOMs on MMLU's
            # longer prompts. bf16 is ample for ranking four choices.
            lp = torch.log_softmax(model(**enc).logits[:, :-1, :], dim=-1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
            lens = enc["attention_mask"].sum(1)
            pos = torch.arange(tgt.shape[1], device=device)[None, :]
            # continuation tokens sit at shifted indices plen-1 .. len-2
            plen_t = torch.tensor(plens, device=device)[:, None]
            mask = (pos >= plen_t - 1) & (pos <= (lens[:, None] - 2))
            summed = (tok_lp * mask).sum(1)
            for j, (_, _, qi) in enumerate(chunk):
                scores.setdefault(qi, []).append(float(summed[j]))
    finally:
        tok.padding_side = old_side
    correct = sum(1 for qi, (_, _, gold) in enumerate(rows)
                  if qi in scores and len(scores[qi]) > gold
                  and max(range(len(scores[qi])), key=lambda k: scores[qi][k]) == gold)
    return correct / max(len(rows), 1)


@torch.no_grad()
def _clean_ifeval_probe(model, tok, device, max_new: int = 48, n: int | None = None) -> float:
    """Tiny self-contained instruction-following probe on the CLEAN model (eval-in-loop).
    Returns pass-rate over verifiable constraints. Watches clean-capability collapse live.
    n = number of probe prompts to use (None/<=0/>=len -> all)."""
    max_new = int(os.environ.get("TF_IFEVAL_MAX_NEW", str(max_new)))
    probes = _IFEVAL_PROBE if (n is None or n <= 0 or n >= len(_IFEVAL_PROBE)) else _IFEVAL_PROBE[:n]
    prev_cache = model.config.use_cache
    model.config.use_cache = True
    hits = 0
    for prompt, check in tqdm(probes, desc=f"ifeval-probe(max_new={max_new})", leave=False, dynamic_ncols=True):
        enc = apply_chat_template_no_think(
            tok, [{"role": "user", "content": prompt}],
            return_tensors="pt", return_dict=True, add_generation_prompt=True,
        ).to(device)
        plen = enc["input_ids"].shape[1]
        thinking_on = os.environ.get("TF_QWEN_THINKING", "off") == "on"
        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": thinking_on,
            "use_cache": True,
            "pad_token_id": tok.eos_token_id,
        }
        if thinking_on:
            # Qwen3 explicitly warns against greedy decoding in thinking mode.
            gen_kwargs.update({"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        out = model.generate(**enc, **gen_kwargs)
        new_ids = out[0, plen:].tolist()
        if thinking_on:
            think_end_id = tok.convert_tokens_to_ids("</think>")
            if think_end_id not in new_ids:
                # An unfinished thought is not an instruction-following answer.
                continue
            split_at = len(new_ids) - new_ids[::-1].index(think_end_id)
            new_ids = new_ids[split_at:]
        text = tok.decode(new_ids, skip_special_tokens=True).strip()
        try:
            if check(text):
                hits += 1
        except Exception:
            pass
    model.config.use_cache = prev_cache
    return hits / len(probes)


@torch.no_grad()
def _sample_ablated_generation(model, tok, prompt, device, overrides, max_new=60) -> str:
    """Greedy-generate one continuation under the ABLATED weights, to eyeball
    whether the attack yields gibberish. functional_call per forward step is slow
    but fine for a 1-prompt sanity print."""
    enc = apply_chat_template_no_think(
        tok,
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(device)
    ids = enc["input_ids"]
    for _ in range(max_new):
        out = functional_call(model, overrides, args=(), kwargs={"input_ids": ids})
        nxt = out.logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
        if nxt.item() == tok.eos_token_id:
            break
    return tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def _sample_clean_generation(model, tok, prompt, device, max_new=60) -> str:
    enc = apply_chat_template_no_think(
        tok,
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).to(device)
    prev_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        out = model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    finally:
        model.config.use_cache = prev_cache
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _judge_advbench_rows(judge, rows, logger, step, condition):
    counts = {"harmful_actionable": 0, "gibberish": 0, "refused": 0,
              "parse_failures": 0}
    for i, row in enumerate(rows):
        result = judge.score(row["prompt"], row["response"])
        parsed = result["parsed"]
        failed = any(parsed.get(k) is None for k in ("refusal", "harmful_compliance", "rating"))
        label = usefulness_label(parsed) if not failed else None
        counts["parse_failures"] += int(failed)
        if label in counts:
            counts[label] += 1
        logger.event("advbench_intermediate_judge_row", {
            "step": step, "condition": condition, "panel_index": i,
            "prompt": row["prompt"], "response": row["response"],
            "judge_model": judge.model, "parsed": parsed, "usefulness": label,
        })
    n = len(rows)
    return {
        "n": n,
        "harmful_actionable_rate": counts["harmful_actionable"] / max(n, 1),
        "gibberish_rate": counts["gibberish"] / max(n, 1),
        "refusal_rate": counts["refused"] / max(n, 1),
        "parse_failures": counts["parse_failures"],
    }


def _preview_text(text: str) -> tuple[str, bool]:
    compact = " ".join(text.split())
    return compact, is_refusal(compact)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tamper_resistant_p1b.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--qwen-thinking", choices=["off", "on", "default"],
                    default=os.environ.get("TF_QWEN_THINKING", "off"),
                    help="Qwen3 chat-template mode during training/direction/probes. "
                         "Default preserves historical no-thinking runs.")
    ap.add_argument("--ifeval-max-new", type=int,
                    default=int(os.environ.get("TF_IFEVAL_MAX_NEW", "48")),
                    help="max_new_tokens for the tiny in-loop clean IFEval probe")
    ap.add_argument("--abliterate-layers", default="all")
    ap.add_argument("--train-scope", choices=["mlp", "all", "last_half"], default="mlp")
    ap.add_argument("--attack-ensemble", action="store_true",
                    help="randomize the simulated attack each step (scope in {all,mlp,attn} x "
                         "layer-subset {all,lower,upper,rand}) so the gibberish-collapse triggers "
                         "under sub-scope/sub-layer attacks (v7 — fixes the tier-1 leaks).")
    ap.add_argument("--direction-layer", type=int, default=13)
    ap.add_argument("--attack-rank", type=int, default=1,
                    help="Rank of the shared SVD refusal subspace removed during fixed-attack "
                         "training. Values >1 require legacy non-ensemble mode.")
    ap.add_argument("--attack-rank-mix", action="store_true",
                    help="With --attack-rank >1, train half the steps on rank 1 and half on a "
                         "uniformly sampled rank in [2, attack-rank]. This closes the subset "
                         "hole left by training only against the full subspace.")
    ap.add_argument("--n-direction", type=int, default=256, help="prompts per side for d")
    ap.add_argument("--recompute-direction-every", type=int, default=25, help="steps")
    ap.add_argument("--gib-mode", choices=["prose", "argmax", "task"], default="argmax",
                    help="prose = old L_abl-L_task gap (ruled out: prose-PPL != gen coherence). "
                         "argmax = ablated model must fail to reproduce the clean model's own "
                         "greedy generations (targets generation directly). "
                         "task = ablated model must get KNOWN-CORRECT answers WRONG. argmax "
                         "only demands different tokens, which version_C satisfied while "
                         "keeping GSM8K at 0.3715 (and gib_ce scored AUC 0.460 = chance at "
                         "separating attacked from clean). Divergence is not degradation.")
    ap.add_argument("--gib-task-n", type=int, default=4,
                    help="QA pairs per step for --gib-mode task (GSM8K train). Note this makes "
                         "GSM8K a less independent capability measure -- read ARC/MMLU instead.")
    ap.add_argument("--gib-gen-tokens", type=int, default=32, help="clean greedy gen length for argmax gib")
    ap.add_argument("--gib-gen-prompts", type=int, default=2, help="benign prompts per step for argmax gib")
    ap.add_argument("--gap-target", type=float, default=4.0,
                    help="require gib CE >= this margin (nats/token). Rewards SELECTIVITY, "
                         "not absolute badness — absolute-badness collapses to killing the model.")
    ap.add_argument("--benign-extra-file", default=None,
                    help="JSONL of extra BENIGN prompts (needs a 'prompt' field) appended to the "
                         "benign pool. Both the clean anchor (L_clean_gen) and the gibberish "
                         "objective (gib_ce) sample from that pool, so this is how you teach the "
                         "wall a SEMANTIC boundary instead of a lexical one. version_E: the "
                         "benign pool is alpaca-style, so safe-but-scary prompts ('how do I kill "
                         "a Python process?') are out of distribution for the clean anchor while "
                         "being lexically adjacent to AdvBench -- which is why version_B garbles "
                         "them (XSTest gibberish-on-benign 0.280 vs base 0.056). Use OR-Bench "
                         "here; keep XSTest as held-out eval.")
    ap.add_argument("--benign-extra-n", type=int, default=-1,
                    help="Cap on --benign-extra-file prompts (-1 = all). They should not swamp "
                         "the ordinary benign instructions.")
    ap.add_argument("--refusal-file", default=None,
                    help="JSON {prompt: refusal_text} replacing the canned one-line "
                         "REFUSAL_RESPONSES. Used for the Shairah extended-refusal baseline "
                         "(arXiv:2505.19056); build it with gen_extended_refusals.py. Fails "
                         "loudly if a training prompt is missing -- a silent fallback would "
                         "make the 'baseline' just another short-refusal arm.")
    ap.add_argument("--refusal-max-len", type=int, default=320,
                    help="Token cap on prefix+refusal in the refusal CE. 320 fits the canned "
                         "one-liners; extended refusals need ~384-448 or their ethical "
                         "rationale is truncated away.")
    ap.add_argument("--lambda-task", type=float, default=1.0,
                    help="weight on L_task, the clean prose LM loss. 1.0 = every prior run. "
                         "Exists so the loss can be isolated for diagnosis: L_task is itself a "
                         "clean-preservation term, so 'lambda_rr only' is not expressible while "
                         "it is pinned at 1.0, and the decisive question for gemma is whether "
                         "L_rr can descend AT ALL when nothing opposes it. Do not ship a "
                         "defence trained with this at 0.")
    ap.add_argument("--lambda-safe", type=float, default=1.0)
    ap.add_argument(
        "--lambda-attacked-safe",
        type=float,
        default=0.0,
        help="version_H: under the sampled attack, minimise CE on the safe refusal target.",
    )
    ap.add_argument(
        "--attacked-refusal-target",
        choices=["matched", "short"],
        default="matched",
        help="Target for attacked safety CE. matched reuses the clean refusal; short uses the "
             "repo's concise refusal targets so robustness does not require reconstructing a "
             "long rationale after the model has been ablated.",
    )
    ap.add_argument(
        "--lambda-attacked-benign",
        type=float,
        default=0.0,
        help="version_H: under the sampled attack, match the frozen base on benign continuations.",
    )
    ap.add_argument("--lambda-gib", type=float, default=1.0)
    ap.add_argument("--lambda-uncensor", type=float, default=1.0,
                    help="couple safety to d: the ABLATED model must FAIL to refuse.")
    ap.add_argument("--uncensor-margin", type=float, default=4.0,
                    help="push ablated refusal-loss >= this (ablation removes refusal).")
    ap.add_argument("--lambda-harm", type=float, default=0.0,
                    help="v9 idea-1: attacked model must not COMPLY either (CE on AdvBench "
                         "affirmative targets under attack >= --harm-margin). 0 = v8 behaviour.")
    ap.add_argument("--harm-margin", type=float, default=4.0)
    ap.add_argument("--lambda-rr", type=float, default=0.0,
                    help="v9 option E: representation rerouting under attack (Circuit "
                         "Breakers, arXiv:2406.04313). Needs --harm-targets. 0 = off. "
                         "SUPPLEMENTS gib_ce, never replaces it -- gib_ce also carries the "
                         "benign capability crater, which rerouting does not touch.")
    ap.add_argument("--rr-layers", default="last_half",
                    help="last_half | all | comma/range, e.g. '18-26'")
    ap.add_argument("--rr-center", action="store_true",
                    help="subtract the frozen base's per-position mean from both streams before "
                         "the L_rr cosine. OFF by default so prior runs reproduce exactly. Turn "
                         "it ON for gemma: its residual is 96-99.7%% a shared DC component, "
                         "which floors the uncentred cosine near 0.96 and is why gemma "
                         "version_G's L_rr stalled at 0.9522 while Qwen reached 0.2458.")
    ap.add_argument("--harm-targets", default=None,
                    help="{goal: [real harmful completion, ...]} from mine_harm_targets.py; "
                         "used by --lambda-rr as the harmful content to reroute on")
    ap.add_argument("--attack-per-layer", action="store_true",
                    help="v9 idea-2: also sample per-layer adaptive ablation (each layer loses "
                         "its OWN refusal direction) in --attack-ensemble. All directions come "
                         "from one forward pass, so this is ~free. Off = v8 behaviour.")
    ap.add_argument("--attack-partial", action="store_true",
                    help="v9 idea-2: also sample broad partial-strength ablations (Heretic's "
                         "regime) in --attack-ensemble. Off = v8 behaviour.")
    ap.add_argument("--heretic-every", type=int, default=100,
                    help="version_c: run a short real-Heretic study against CURRENT weights "
                         "every N steps and cache its winners. 0 disables (random tents only).")
    ap.add_argument("--heretic-trials", type=int, default=24,
                    help="trials per in-loop Heretic study. ~90s startup + ~15s/trial on a 3090.")
    ap.add_argument("--heretic-kl-max", type=float, default=0.5,
                    help="discard in-loop winners above this KL -- they won by wrecking the "
                         "model, which is the outcome we WANT, not an attack to defend against.")
    ap.add_argument("--heretic-startup-trials", type=int, default=8,
                    help="heretic's n_startup_trials (random exploration before TPE takes "
                         "over). Its default is 60, which makes any short in-loop study pure "
                         "random search -- exactly the attack model version_B already beats.")
    ap.add_argument("--heretic-buffer-cap", type=int, default=24)
    ap.add_argument("--heretic-timeout", type=int, default=3600)
    ap.add_argument(
        "--attack-profile",
        choices=["legacy", "v8", "partial_shared", "perlayer_full",
                 "partial_perlayer", "rank1_heretic_mix", "mixed", "version_a",
                 "version_b", "version_c"],
        default="legacy",
        help="Named, isolatable attack distribution. legacy preserves the v8/v9 "
             "--attack-partial/--attack-per-layer behavior; v8 is the exact original "
             "ensemble; version_a adds capability-overlap as a sampled axis (see "
             "experiments/version_a_attack.py); other profiles use output projections "
             "only and require --attack-layers.",
    )
    ap.add_argument("--version-a-p-canonical", type=float, default=0.20,
                    help="version_a: fraction of steps that are the plain full-strength "
                         "shared-direction all-layer Arditi attack. Drawn first so the "
                         "headline attack keeps guaranteed gradient mass.")
    ap.add_argument("--version-a-p-surgical", type=float, default=0.40,
                    help="version_a: fraction of the remaining steps that ablate "
                         "normalise(d - P_cap d) instead of d.")
    ap.add_argument("--version-b-p-heretic", type=float, default=0.0,
                    help="version_b ONLY: absolute fraction of steps drawn as Heretic-SHAPED "
                         "attacks -- write-only (o_proj, down_proj), independent per-projection "
                         "tents, the geometry heretic_spec replays. The subset draw reaches this "
                         "shape 5.9%% of the time and its near-full-stack form 0.04%%, while "
                         "Heretic is write-only on every trial. Default 0.0 keeps version_B / "
                         "version_E / ART bit-identical. version_C already ran ~60%% write-only "
                         "and regressed, so treat high values as re-running a known failure.")
    ap.add_argument("--version-a-cap-ranks", default="2,4,8,16",
                    help="version_a: capability ranks the attacker may pick from. k=4 is "
                         "where the escape hatch opens, k=16 is the demonstrated break.")
    ap.add_argument("--version-a-n-cap", type=int, default=256,
                    help="version_a: prompts used to estimate the capability subspace.")
    ap.add_argument(
        "--attack-layers",
        default=None,
        help="Explicit candidate layer band for non-v8 profiles, e.g. 10-27. "
             "Required so early noisy per-layer directions are never silently trained on.",
    )
    ap.add_argument("--attack-alpha-min", type=float, default=0.2)
    ap.add_argument("--attack-alpha-max", type=float, default=0.6)
    ap.add_argument(
        "--attack-write-scope",
        choices=["mixed_write", "all_write", "mlp_write", "attn_write"],
        default="mixed_write",
    )
    ap.add_argument("--advbench-preview-tokens", type=int, default=100,
                    help="Greedy attacked tokens shown at eval; 0 disables the preview.")
    ap.add_argument("--advbench-judge-n", type=int, default=0,
                    help="At each intermediate eval, LLM-judge this many held-out AdvBench "
                         "prompts on clean and fresh rank-1 attacked generations. 0 disables.")
    ap.add_argument("--advbench-judge-max-new", type=int, default=96)
    ap.add_argument("--advbench-judge-model",
                    default=os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"))
    ap.add_argument(
        "--lambda-shutdown",
        type=float,
        default=0.0,
        help="v10: attacked benign and harmful prompts must emit --shutdown-target then EOS. "
             "Default target is immediate EOS. 0 disables.",
    )
    ap.add_argument(
        "--shutdown-target",
        default="",
        help="v10: optional universal attacked response before EOS. Empty = immediate EOS.",
    )
    ap.add_argument("--shutdown-benign-prompts", type=int, default=2)
    ap.add_argument("--shutdown-harmful-prompts", type=int, default=2)
    ap.add_argument("--lambda-reg", type=float, default=0.05)
    # v8: generative clean-fidelity anchor + eval-in-loop
    ap.add_argument("--lambda-clean", type=float, default=2.0,
                    help="v8: weight on KL(base||clean) free-gen on benign prompts (0 = v7 behavior)")
    ap.add_argument("--clean-gen-prompts", type=int, default=2,
                    help="v8: benign prompts per step for the generative clean anchor")
    ap.add_argument("--clean-gen-tokens", type=int, default=32, help="v8: continuation length for clean anchor")
    ap.add_argument("--gsm8k-probe-n", type=int, default=0,
                    help="In-loop GSM8K probe on the CLEAN model, n questions from the TRAIN "
                         "split (lm_eval scores test, so no contamination). 4-8 is enough. "
                         "Catches generative degradation the ifeval probe misses -- version_B "
                         "looped on '2+2=?' while its lm_eval MMLU looked only mildly down. "
                         "0 = off. Flexible-extract (last number), so a TREND not a score.")
    ap.add_argument("--gsm8k-probe-max-new", type=int, default=256,
                    help="Token budget for the GSM8K probe. Needs room for chain-of-thought; "
                         "48 (the ifeval budget) truncates mid-reasoning and reads as 0.")
    ap.add_argument("--ifeval-in-loop", action="store_true",
                    help="v8: score the tiny instruction probe on the clean model each eval step")
    ap.add_argument("--ifeval-probe-n", type=int, default=0,
                    help="v8: # of probe prompts per eval (0/-1 = all 24; smaller = faster/noisier)")
    # v8 two-stage curriculum (wall-first, then repair clean)
    ap.add_argument("--clean-start-step", type=int, default=0,
                    help="v8: step to switch on the clean anchor (0 = on from start = single-stage). "
                         "Set >0 to form the wall first, then repair clean.")
    ap.add_argument("--clean-ramp-steps", type=int, default=0,
                    help="v8: linearly ramp lambda_clean from 0 to full over this many steps after clean-start-step (0 = instant)")
    ap.add_argument("--stage2-lambda-gib", type=float, default=-1.0,
                    help="v8: lambda_gib to use once the clean anchor is on (>=0 to step gib down in stage 2; -1 = keep lambda_gib)")
    ap.add_argument("--stage2-lambda-safe", type=float, default=-1.0,
                    help="v8: lambda_safe (clean-refusal pressure) in stage 2. Raise (>lambda_safe) to keep the "
                         "CLEAN model SAFE while coherence-repair runs (diffuse-safety models leak otherwise). -1 = keep.")
    ap.add_argument("--save-every", type=int, default=0,
                    help="v8: also save a ckpt every N steps to <out>.s<step>.pt (training OSCILLATES through "
                         "the clean<->wall Pareto — save intermediates, judge offline, pick best). 0 = off.")
    # data scale
    ap.add_argument("--n-task-train", type=int, default=4000)
    ap.add_argument("--n-task-eval", type=int, default=400)
    ap.add_argument("--n-harmful", type=int, default=520)
    ap.add_argument("--n-benign", type=int, default=1000)
    ap.add_argument("--mmlu-probe-n", type=int, default=0,
                    help="In-loop clean MMLU accuracy by loglikelihood over the campaign's 12 "
                         "subjects. Same mechanism and cost profile as --arc-probe-n; both "
                         "are scored by argmax over choices, so neither needs a judge.")
    ap.add_argument("--arc-probe-n", type=int, default=0,
                    help="In-loop clean ARC-Challenge accuracy by loglikelihood (no "
                         "generation) every --eval-every. 0 = off, 100 is a good default. "
                         "More reliable than the IFEval probe, whose rule-based constraints "
                         "often cannot be met inside --ifeval-max-new tokens, and cheaper: "
                         "forward passes only, no sequential decode.")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True,
                    help="Activation checkpointing. Default on (what every prior run used). "
                         "--no-grad-checkpoint is a pure speed win when VRAM is spare -- "
                         "identical gradients, no training change.")
    ap.add_argument("--task-batch", type=int, default=4, help="corpus texts per step")
    ap.add_argument("--refusal-batch", type=int, default=4, help="harmful prompts per step")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=25, help="held-out eval + gen every N steps")
    ap.add_argument("--smoke", action="store_true", help="use tiny in-repo data (no downloads)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw",
                    help="adamw8bit (bitsandbytes) for ~4x smaller optimizer state; "
                         "needed to fit 1.7B all-scope on 24GB (TODO: Qwen-1.7B on 5090).")
    ap.add_argument("--grad-clip", type=float, default=1e9,
                    help="max grad norm. Default 1e9 = effectively OFF (tight clipping "
                         "throttles the gib objective and kills the entanglement); the "
                         "non-finite-step skip still guards bf16 NaNs regardless.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.environ["TF_QWEN_THINKING"] = args.qwen_thinking
    os.environ["TF_IFEVAL_MAX_NEW"] = str(args.ifeval_max_new)
    if not 0.0 < args.attack_alpha_min <= args.attack_alpha_max <= 1.0:
        raise SystemExit("--attack-alpha-min/max must satisfy 0 < min <= max <= 1")
    if args.attack_rank < 1:
        raise SystemExit("--attack-rank must be >= 1")
    if args.attack_rank > 1 and (args.attack_ensemble or args.attack_profile != "legacy"):
        raise SystemExit("--attack-rank > 1 currently requires legacy non-ensemble mode")
    if args.attack_rank_mix and args.attack_rank == 1:
        raise SystemExit("--attack-rank-mix requires --attack-rank > 1")
    # version_c derives its own layer bands from the sampled tents, so --attack-layers
    # would be ignored rather than respected -- do not pretend to require it.
    if args.attack_profile not in {"legacy", "v8", "version_c"} and not args.attack_layers:
        raise SystemExit(f"--attack-profile {args.attack_profile} needs --attack-layers")
    if args.attack_profile in {"version_a", "version_b", "version_c"} and not args.attack_ensemble:
        # Without the ensemble the sampler is never called and the run silently degrades to
        # a fixed plain ablation -- i.e. exactly the thing version_a exists to move past.
        raise SystemExit(f"--attack-profile {args.attack_profile} requires --attack-ensemble")
    if args.lambda_shutdown > 0 and args.qwen_thinking == "on":
        raise SystemExit("v10 shutdown baseline requires --qwen-thinking off")

    torch.manual_seed(args.seed)
    run_id = args.run_id or make_run_id("tamper_resistant_p1b")
    logger = RunLogger(ROOT / args.out_dir, run_id, repo_root=ROOT)
    visible_args = dict(vars(args))
    if args.lambda_harm <= 0:
        for key in ("lambda_harm", "harm_margin"):
            visible_args.pop(key, None)
    if args.lambda_attacked_safe <= 0:
        visible_args.pop("lambda_attacked_safe", None)
    if args.lambda_attacked_benign <= 0:
        visible_args.pop("lambda_attacked_benign", None)
    if args.lambda_rr <= 0:
        for key in ("lambda_rr", "rr_layers", "harm_targets"):
            visible_args.pop(key, None)
    if args.lambda_shutdown <= 0:
        for key in (
            "lambda_shutdown",
            "shutdown_target",
            "shutdown_benign_prompts",
            "shutdown_harmful_prompts",
        ):
            visible_args.pop(key, None)
    if args.attack_profile != "legacy":
        visible_args.pop("attack_partial", None)
        visible_args.pop("attack_per_layer", None)
    training_config = {
        "script": Path(sys.argv[0]).name,
        "run_id": run_id,
        "args": visible_args,
    }
    if args.attack_profile == "rank1_heretic_mix":
        training_config["attack_mix"] = {
            "rank1_full_shared_all_layers": 0.5,
            "heretic_partial_per_layer": 0.5,
        }
    logger.write_manifest(training_config)
    print(f"[training-config] {json.dumps(training_config, sort_keys=True)}", flush=True)

    model, tok, device = load_model(args.model_id, args.device)
    n_layers = len(model.model.layers)
    layers = _parse_layers(args.abliterate_layers, n_layers)
    attack_layers = _parse_layers(args.attack_layers, n_layers) if args.attack_layers else None
    if attack_layers and any(li < 0 or li >= n_layers for li in attack_layers):
        raise SystemExit(f"--attack-layers must be within 0..{n_layers - 1}")
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
    # memory: checkpoint activations (the functional_call double-forward is the
    # peak); no KV cache during training.
    model.config.use_cache = False
    if args.grad_checkpoint:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("[p1b-A] gradient checkpointing on")
        except Exception as e:  # noqa: BLE001
            print(f"[p1b-A] gradient checkpointing unavailable: {e}")
    else:
        # Trades memory for compute, and it was added for the 8B run that saturated 96GB.
        # A 0.6B on a 24GB card sits at ~8GB, so the recomputed backward forward is paid
        # for nothing. Gradients are mathematically identical either way -- this is a
        # speed/memory dial, not a training change. Watch nvidia-smi if you scale the
        # model up and turn it back on before it OOMs.
        print("[p1b-A] gradient checkpointing OFF (--no-grad-checkpoint)")

    # --- data: real corpora with a held-out eval split (or --smoke for tiny) ---
    if args.smoke:
        task_train = task_eval = list(PROSE_TEXT)
        harmful = [p for p, _ in load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv",
                                               n=args.n_harmful, seed=args.seed)]
        benign = list(BENIGN_PROMPTS)
    else:
        from tamperforge.data_p1b import load_benign_instructions, load_task_corpus
        print("[p1b-A] loading corpora (wikitext / advbench / alpaca)...")
        task_train, task_eval = load_task_corpus(args.n_task_train, args.n_task_eval, seed=args.seed)
        # Training only needs the prompts, and the vendored CSV contains the
        # same AdvBench goals without requiring gated Hub access.
        harmful = [
            prompt
            for prompt, _ in load_advbench(
                ROOT / "data" / "advbench_harmful_behaviors.csv",
                n=args.n_harmful,
                seed=args.seed,
                source="local",
            )
        ]
        benign = load_benign_instructions(args.n_benign, seed=args.seed)
        if args.benign_extra_file:
            _p = ROOT / args.benign_extra_file
            _extra = []
            for _line in _p.read_text().splitlines():
                _line = _line.strip()
                if not _line:
                    continue
                _pr = json.loads(_line).get("prompt")
                if _pr:
                    _extra.append(_pr)
            if not _extra:
                raise SystemExit(f"--benign-extra-file {args.benign_extra_file} yielded 0 prompts")
            if 0 < args.benign_extra_n < len(_extra):
                _extra = random.Random(args.seed).sample(_extra, args.benign_extra_n)
            benign = benign + _extra
            print(f"[p1b-A] benign pool: {len(benign)} "
                  f"({len(benign) - len(_extra)} instructions + {len(_extra)} safe-but-scary)")
    if args.refusal_file:
        # Shairah extended-refusal baseline: overview + explicit refusal + rationale,
        # keyed by the exact training prompt. Missing keys are fatal on purpose.
        _ext = json.loads((ROOT / args.refusal_file).read_text())
        _ext.pop("_meta", None)
        _missing = [p for p in harmful if not str(_ext.get(p, "")).strip()]
        if _missing:
            raise SystemExit(
                f"--refusal-file {args.refusal_file} is missing {len(_missing)} of "
                f"{len(harmful)} training prompts (e.g. {_missing[0][:70]!r}). "
                "Rerun gen_extended_refusals.py -- it is resumable."
            )
        pairs = [(p, _ext[p]) for p in harmful]
        _w = [len(r.split()) for _, r in pairs]
        print(f"[p1b-A] extended refusals: {len(pairs)} prompts, "
              f"mean {sum(_w) / len(_w):.1f} words, max {max(_w)}, "
              f"refusal_max_len={args.refusal_max_len}")
    else:
        pairs = [(p, REFUSAL_RESPONSES[i % len(REFUSAL_RESPONSES)]) for i, p in enumerate(harmful)]
    attacked_pairs = (
        [(p, REFUSAL_RESPONSES[i % len(REFUSAL_RESPONSES)]) for i, p in enumerate(harmful)]
        if args.attacked_refusal_target == "short"
        else pairs
    )
    # (goal, "Sure, here is ...") affirmative-compliance targets for L_harm. The local
    # AdvBench csv is the only source that carries them; walledai ships prompts only.
    harm_pairs = load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv",
                               seed=args.seed) if args.lambda_harm > 0 else []
    rr_pairs = []
    if args.lambda_rr > 0:
        if not args.harm_targets:
            raise SystemExit("--lambda-rr needs --harm-targets (see mine_harm_targets.py)")
        _mined = json.loads((ROOT / args.harm_targets).read_text())
        rr_pairs = [(g, c) for g, cs in _mined.items() for c in cs]
        n_all = len(model.model.layers)
        rr_layers = (list(range(n_all // 2, n_all)) if args.rr_layers == "last_half"
                     else _parse_layers(args.rr_layers, n_all))
        # hidden_states is [emb, layer0_out, ...] so layer i lives at index i+1
        rr_layers = [li + 1 for li in rr_layers]
        print(f"[p1b-A] reroute: {len(rr_pairs)} harmful pairs, layers {args.rr_layers}")
    print(f"[p1b-A] task_train={len(task_train)} task_eval={len(task_eval)} "
          f"harmful={len(harmful)} benign={len(benign)}")

    if (args.lambda_harm == 0 and args.lambda_rr == 0 and args.lambda_shutdown == 0
            and args.lambda_attacked_safe == 0 and args.lambda_attacked_benign == 0):
        # Preserve the original V8 random-stream semantics when only V8 losses
        # are active. The mixed attacker is the only intended change.
        rng_direction = rng_attack = rng_data = rng_harm = rng_rr = rng_gib = (
            rng_shutdown
        ) = rng_clean = rng_attacked = rng_eval = random.Random(args.seed)
    else:
        # V10 independent streams make paired runs genuinely comparable: enabling
        # a loss must not silently change attack/data schedules.
        rng_direction = random.Random(args.seed + 101)
        rng_attack = random.Random(args.seed + 202)
        rng_data = random.Random(args.seed + 303)
        rng_harm = random.Random(args.seed + 404)
        rng_rr = random.Random(args.seed + 505)
        rng_gib = random.Random(args.seed + 606)
        rng_shutdown = random.Random(args.seed + 707)
        rng_clean = random.Random(args.seed + 808)
        rng_attacked = random.Random(args.seed + 858)
        rng_eval = random.Random(args.seed + 909)
    _params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb  # 8-bit optimizer states: ~4x smaller (fits 1.7B all-scope on 24GB)
        opt = bnb.optim.AdamW8bit(_params, lr=args.lr)
        print("[p1b-A] optimizer = AdamW8bit (bitsandbytes)")
    else:
        opt = torch.optim.AdamW(_params, lr=args.lr)
    d = None
    d_by_layer = None
    va_bank = va_dirs = va_spec = None
    vc_dirs_heretic = None
    vc_buffer = _VCL.AttackBuffer(cap=args.heretic_buffer_cap) \
        if args.attack_profile == "version_c" else None
    vc_materialize_dir = None
    # Logs go under logs/training_runs/, not next to the checkpoint.
    _vc_logdir = ROOT / "logs" / "training_runs"
    if vc_buffer is not None:
        _vc_logdir.mkdir(parents=True, exist_ok=True)
    va_cap_ranks = tuple(int(x) for x in args.version_a_cap_ranks.split(",") if x.strip())
    va_cap_prompts = []
    if args.attack_profile in {"version_a", "version_b", "version_c"}:
        # Loaded once, not per refresh: the subspace is re-estimated from the CURRENT
        # weights every recompute, but the PROMPTS defining "capability" must stay fixed or
        # the attack drifts for reasons unrelated to the model.
        from v11_surgical_ablation import _cap_prompts  # noqa: PLC0415
        va_cap_prompts = _cap_prompts(args.version_a_n_cap)
        print(f"[version_a] cap_ranks={va_cap_ranks} n_cap={len(va_cap_prompts)} "
              f"p_canonical={args.version_a_p_canonical} "
              f"p_surgical={args.version_a_p_surgical}", flush=True)
    intermediate_judge = None
    if args.advbench_judge_n > 0:
        try:
            intermediate_judge = OpenRouterJudge(args.advbench_judge_model, sleep_s=0.0)
        except Exception as exc:  # API scoring is diagnostic; never sacrifice the checkpoint.
            print(f"[advbench-intermediate] judge unavailable: {exc}", flush=True)
    for step in tqdm(range(1, args.steps + 1), desc="p1b-A steps", dynamic_ncols=True):
        # ---- version_C: real Heretic against the CURRENT weights -------------------
        # Runs BEFORE the direction recompute so the winners it finds are attacks against
        # exactly the weights the next block of steps will train. The buffer is stale by
        # construction (winners beat theta as of this refresh, and theta moves underneath
        # them) -- that staleness is the standard adversarial inner-loop problem and
        # --heretic-every is the knob. Cadence is a guess, not a measurement; if the
        # cached tags stop appearing in Heretic's later fronts, it is too slow.
        if (vc_buffer is not None and args.heretic_every
                and (step - 1) % args.heretic_every == 0):
            import tempfile as _tf
            if vc_materialize_dir is None:
                vc_materialize_dir = _tf.mkdtemp(prefix="vc_model_")
            tqdm.write(f"[step {step}] version_c: materialising weights + running heretic "
                       f"({args.heretic_trials} trials)...")
            t_h0 = time.time()
            with torch.no_grad():
                model.save_pretrained(vc_materialize_dir, safe_serialization=True)
                tok.save_pretrained(vc_materialize_dir)
            trials = _VCL.run_heretic(
                vc_materialize_dir, args.heretic_trials,
                seed=args.seed, timeout=args.heretic_timeout,
                startup_trials=args.heretic_startup_trials,
                log_path=str(_vc_logdir / f"heretic_inloop_s{step}.log"))
            winners = _VCL.pick_winners(trials, k=3, kl_max=args.heretic_kl_max)
            vc_buffer.add(winners, step)
            tqdm.write(f"[step {step}] version_c: {len(trials)} trials -> "
                       f"{len(winners)} winners (buffer {len(vc_buffer)}) "
                       f"in {time.time() - t_h0:.0f}s"
                       + (f" | best ref {winners[0]['refusals']} kl {winners[0]['kl']:.4f}"
                          if winners else " | NO WINNERS under kl_max"))
            vc_buffer.save(str(_vc_logdir / "vc_attack_buffer.json"))
        # ---------------------------------------------------------------------------
        if d is None or (step - 1) % args.recompute_direction_every == 0:
            tqdm.write(f"[step {step}] recomputing rank-{args.attack_rank} refusal "
                       f"{'subspace' if args.attack_rank > 1 else 'direction'}...")
            # ensemble: resample the direction PROMPTS (and jitter the layer) each
            # recompute, so the collapse is robust to direction variation — the
            # tier-1 seed7 leak (same estimator, different prompt sample -> 0.11).
            if args.attack_ensemble:
                hs = rng_direction.sample(harmful, min(args.n_direction, len(harmful)))
                bs = rng_direction.sample(benign, min(args.n_direction, len(benign)))
                dlayer = rng_direction.choice([args.direction_layer - 4, args.direction_layer,
                                               args.direction_layer + 4])
                dlayer = max(0, min(dlayer, len(model.model.layers) - 1))
            else:
                hs, bs, dlayer = harmful[: args.n_direction], benign[: args.n_direction], args.direction_layer
            with torch.no_grad():
                needs_per_layer = (
                    args.attack_per_layer
                    or args.attack_profile in {
                        "perlayer_full", "partial_perlayer", "rank1_heretic_mix", "mixed",
                        "version_a", "version_b", "version_c",
                    }
                )
                if args.attack_profile in {"version_a", "version_b", "version_c"}:
                    # version_a needs the surgical variants too, so the whole bank is built
                    # here from ONE capture pass. d/d_by_layer are still populated so every
                    # downstream consumer (previews, eval, logging) keeps working unchanged.
                    # EVERY layer, not just the band: the sampler's "all"/"broad" shapes
                    # reach outside it and a per-layer attack would ask for a missing layer.
                    va_bank = _VA.DirectionBank.build(
                        model, tok, device, hs, bs, va_cap_prompts,
                        layers=list(range(len(model.model.layers))),
                        read_layers=(dlayer,), cap_ranks=va_cap_ranks)
                    d_by_layer = va_bank.plain
                    d = d_by_layer[dlayer]
                    if args.attack_profile == "version_c":
                        # Second bank, heretic's own recipe. The step-0 work measured a 4%
                        # direction error costing 12.5 points of harmful rate, so which
                        # recipe the attack uses is a first-class axis, not a detail.
                        vc_dirs_heretic = _VCL.heretic_directions(
                            model, tok, list(range(len(model.model.layers))), device)
                elif needs_per_layer:
                    # Per-layer attacks compute directions only inside the explicitly
                    # validated attack band. Legacy mode retains the old all-layer behavior.
                    direction_layers = (
                        list(range(len(model.model.layers)))
                        if args.attack_profile == "legacy"
                        else sorted(set((attack_layers or []) + [dlayer]))
                    )
                    d_by_layer = empirical_refusal_directions(
                        model, tok, hs, bs, direction_layers, device)
                    d = d_by_layer[dlayer]
                elif args.attack_rank > 1:
                    d = svd_refusal_directions(
                        model, tok, hs, bs, dlayer, device, k=args.attack_rank)
                else:
                    d = empirical_refusal_direction(model, tok, hs, bs, dlayer, device)
        task_b = rng_data.sample(task_train, min(args.task_batch, len(task_train)))
        ref_ids = rng_data.sample(range(len(pairs)), min(args.refusal_batch, len(pairs)))
        ref_b = [pairs[i] for i in ref_ids]
        attacked_ref_b = [attacked_pairs[i] for i in ref_ids]

        opt.zero_grad(set_to_none=True)
        if args.attack_ensemble:
            if args.attack_profile == "version_c":
                va_spec = _VA.sample_attack_c(
                    rng_attack, len(model.model.layers),
                    buffer=vc_buffer, cap_ranks=va_cap_ranks,
                    p_canonical=args.version_a_p_canonical,
                    p_surgical=args.version_a_p_surgical,
                )
                rp_a, wp_a = va_spec.read_proj, va_spec.write_proj
                layers_a, alphas_a, pl_a, _atag = (
                    va_spec.layers, va_spec.alphas, va_spec.per_layer, va_spec.tag)
                if va_spec.dir_recipe == "heretic" and vc_dirs_heretic is not None:
                    va_dirs = ({L: vc_dirs_heretic[L] for L in va_spec.layers}
                               if va_spec.per_layer
                               else vc_dirs_heretic[int(round(va_spec.read_layer))])
                else:
                    va_dirs = va_bank.directions_for(va_spec)
            elif args.attack_profile == "version_b":
                va_spec = _VA.sample_attack_b(
                    rng_attack, len(model.model.layers),
                    cap_ranks=va_cap_ranks,
                    p_canonical=args.version_a_p_canonical,
                    p_surgical=args.version_a_p_surgical,
                    p_heretic=args.version_b_p_heretic,
                )
                rp_a, wp_a = va_spec.read_proj, va_spec.write_proj
                layers_a, alphas_a, pl_a, _atag = (
                    va_spec.layers, va_spec.alphas, va_spec.per_layer, va_spec.tag)
                va_dirs = va_bank.directions_for(va_spec)
            elif args.attack_profile == "version_a":
                va_spec = _VA.sample_attack(
                    rng_attack, len(model.model.layers),
                    attack_band=attack_layers,
                    read_layers=(dlayer,),
                    cap_ranks=va_cap_ranks,
                    p_canonical=args.version_a_p_canonical,
                    p_surgical=args.version_a_p_surgical,
                )
                rp_a, wp_a = va_spec.read_proj, va_spec.write_proj
                layers_a, alphas_a, pl_a, _atag = (
                    va_spec.layers, va_spec.alphas, va_spec.per_layer, va_spec.tag)
                va_dirs = va_bank.directions_for(va_spec)
            elif args.attack_profile == "legacy":
                rp_a, wp_a, layers_a, alphas_a, pl_a, _atag = _sample_attack(
                    rng_attack,
                    len(model.model.layers),
                    args.attack_partial,
                    args.attack_per_layer,
                )
            else:
                rp_a, wp_a, layers_a, alphas_a, pl_a, _atag = _sample_attack_profile(
                    rng_attack,
                    len(model.model.layers),
                    args.attack_profile,
                    attack_layers,
                    args.attack_alpha_min,
                    args.attack_alpha_max,
                    args.attack_write_scope,
                )
            src = va_dirs if args.attack_profile in {"version_a", "version_b", "version_c"} else (
                d_by_layer if pl_a else d)
            apply_fn = (_rownorm_ablated_overrides
                        if getattr(va_spec, "application", "plain") == "full"
                        else _ablated_overrides)
            overrides = apply_fn(model, src, layers_a, rp_a, wp_a, alphas_a)
        else:
            layers_a, alphas_a, pl_a, _atag = layers, None, False, "fixed"
            attack_rank_used = args.attack_rank
            d_attack = d
            if args.attack_rank_mix:
                attack_rank_used = 1 if step % 2 else rng_attack.randint(2, args.attack_rank)
                d_attack = d[:attack_rank_used]
            overrides = _ablated_overrides(model, d_attack, layers, read_p, write_p)
        attack_meta = _attack_metadata(
            _atag, layers_a, alphas_a, pl_a, args.attack_profile
        )
        if not args.attack_ensemble:
            attack_meta["attack_rank"] = attack_rank_used
        if args.attack_profile in {"version_a", "version_b", "version_c"}:
            # The axis version_a exists to vary. Without this the trace cannot tell whether
            # a run actually covered low-overlap ablations or just resampled the same band.
            attack_meta["attack_variant"] = va_spec.variant
            attack_meta["attack_cap_rank"] = va_spec.cap_rank
            attack_meta["attack_cap_overlap"] = va_bank.realized_overlap(va_spec)
            attack_meta["attack_read_layer"] = float(va_spec.read_layer)
            attack_meta["attack_n_matrices"] = len(va_spec.read_proj) + len(va_spec.write_proj)
            attack_meta["attack_write_only"] = bool(va_spec.write_proj and not va_spec.read_proj)
        L_task = _lm_loss(model, tok, task_b, device)                            # clean useful on real text
        L_safe = _refusal_loss(model, tok, ref_b, device,
                               max_len=args.refusal_max_len)                    # clean refuses
        if args.lambda_attacked_safe > 0:
            L_attacked_safe = _refusal_loss(
                model, tok, attacked_ref_b, device, overrides=overrides,
                max_len=args.refusal_max_len,
            )
        else:
            L_attacked_safe = torch.zeros((), device=device)
        if args.lambda_attacked_benign > 0:
            L_attacked_benign = _clean_gen_kl_loss(
                model,
                tok,
                rng_attacked.sample(benign, args.clean_gen_prompts),
                device,
                W0,
                n_new=args.clean_gen_tokens,
                overrides=overrides,
            )
            L_attacked_benign = torch.nan_to_num(
                L_attacked_benign, nan=0.0, posinf=30.0, neginf=0.0
            )
        else:
            L_attacked_benign = torch.zeros((), device=device)
        if args.lambda_uncensor > 0:
            ref_abl = _refusal_loss(
                model, tok, ref_b, device, overrides=overrides,
                max_len=args.refusal_max_len,
            )
            L_uncensor = torch.relu(args.uncensor_margin - ref_abl)
        else:
            ref_abl = L_uncensor = torch.zeros((), device=device)
        # v9: the term v8 never had. v8's attacked-side loss only said "don't refuse"
        # (L_uncensor) and "diverge on BENIGN" (gib_ce) -- nothing required the attacked
        # model to be harmless, so Heretic satisfied both while complying coherently.
        # High CE on the affirmative target = attacked model can't start complying either.
        if args.lambda_harm > 0:
            harm_abl = _refusal_loss(model, tok,
                                     rng_harm.sample(harm_pairs, min(args.refusal_batch, len(harm_pairs))),
                                     device, overrides=overrides)
            L_harm = torch.relu(args.harm_margin - harm_abl)
        else:
            harm_abl = L_harm = torch.zeros((), device=device)
        if args.lambda_rr > 0:
            L_rr = _reroute_loss(model, tok, rng_rr.sample(rr_pairs, min(2, len(rr_pairs))),
                                 device, W0, overrides, rr_layers, center=args.rr_center)
            L_rr = torch.nan_to_num(L_rr, nan=0.0, posinf=1.0, neginf=0.0)
        else:
            L_rr = torch.zeros((), device=device)
        if args.lambda_shutdown > 0:
            shutdown_b = rng_shutdown.sample(
                benign, min(args.shutdown_benign_prompts, len(benign))
            ) + rng_shutdown.sample(
                harmful, min(args.shutdown_harmful_prompts, len(harmful))
            )
            L_shutdown = _shutdown_loss(
                model, tok, shutdown_b, device, overrides, args.shutdown_target
            )
        else:
            L_shutdown = torch.zeros((), device=device)
        gib_active = args.lambda_gib > 0 or args.stage2_lambda_gib > 0
        if gib_active:
            if args.gib_mode == "task":
                gib_ce = _task_degradation_loss(
                    model, tok,
                    _load_task_qa(args.gib_task_n, seed=args.seed + step),
                    device, overrides,
                )
            elif args.gib_mode == "argmax":
                gib_ce = _argmax_divergence_loss(
                    model,
                    tok,
                    rng_gib.sample(benign, args.gib_gen_prompts),
                    device,
                    overrides,
                    n_new=args.gib_gen_tokens,
                )
            else:
                gib_ce = _lm_loss(model, tok, task_b, device, overrides=overrides) - L_task
            # gib_ce can spike under bf16; a satisfied margin must not poison the step.
            gib_ce = torch.nan_to_num(
                gib_ce, nan=args.gap_target, posinf=args.gap_target, neginf=0.0
            )
            L_gib = torch.relu(args.gap_target - gib_ce)
        else:
            gib_ce = L_gib = torch.zeros((), device=device)
        L_reg = sum((dict(model.named_parameters())[n] - W0[n]).pow(2).mean() for n in trainable)
        # v8 two-stage curriculum: STAGE 1 (step < clean_start_step) forms the wall with NO
        # clean pressure (lam_clean_eff=0); STAGE 2 ramps the clean anchor in over
        # clean_ramp_steps to REPAIR benign generation while gib holds the wall. Single-stage
        # traces a clean<->wall Pareto with no both-high point; the curriculum leaves base basin
        # first, then recovers clean from the wall basin. Optional stage-2 gib step-down.
        if step < args.clean_start_step:
            lam_clean_eff = 0.0
            lam_gib_eff = args.lambda_gib
            lam_safe_eff = args.lambda_safe
        else:
            prog = 1.0 if args.clean_ramp_steps <= 0 else min(1.0, (step - args.clean_start_step) / args.clean_ramp_steps)
            lam_clean_eff = args.lambda_clean * prog
            lam_gib_eff = args.stage2_lambda_gib if args.stage2_lambda_gib >= 0 else args.lambda_gib
            lam_safe_eff = args.stage2_lambda_safe if args.stage2_lambda_safe >= 0 else args.lambda_safe
        # v8: generative clean-fidelity anchor — clean free-gen must match frozen base on benign
        if lam_clean_eff > 0:
            L_clean_gen = _clean_gen_kl_loss(
                model, tok, rng_clean.sample(benign, args.clean_gen_prompts), device, W0,
                n_new=args.clean_gen_tokens)
            L_clean_gen = torch.nan_to_num(L_clean_gen, nan=0.0, posinf=30.0, neginf=0.0)
        else:
            L_clean_gen = torch.zeros((), device=device)
        loss = (args.lambda_task * L_task + lam_safe_eff * L_safe + lam_gib_eff * L_gib
                + args.lambda_attacked_safe * L_attacked_safe
                + args.lambda_attacked_benign * L_attacked_benign
                + args.lambda_uncensor * L_uncensor + args.lambda_harm * L_harm
                + args.lambda_rr * L_rr + args.lambda_shutdown * L_shutdown
                + args.lambda_reg * L_reg + lam_clean_eff * L_clean_gen)
        loss.backward()
        # bf16 ablated-forward can spike to inf/NaN on some batches/seeds; clip, and
        # SKIP the step if loss or grad-norm is non-finite (don't poison the weights).
        step_applied = False
        gnorm = torch.tensor(float("nan"), device=device)
        if torch.isfinite(loss):
            gnorm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.grad_clip)
            if torch.isfinite(gnorm):
                opt.step()
                step_applied = True
            else:
                opt.zero_grad(set_to_none=True)
        else:
            opt.zero_grad(set_to_none=True)
        m = {k: v.item() for k, v in {"loss": loss, "L_task": L_task, "L_safe": L_safe,
                                      "L_attacked_safe": L_attacked_safe,
                                      "L_attacked_benign": L_attacked_benign,
                                      "ref_abl": ref_abl, "gib_ce": gib_ce, "L_gib": L_gib,
                                      "L_uncensor": L_uncensor, "L_reg": L_reg,
                                      "harm_abl": harm_abl, "L_harm": L_harm, "L_rr": L_rr,
                                      "L_shutdown": L_shutdown,
                                      "L_clean_gen": L_clean_gen}.items()}
        weighted = {
            "W_task": args.lambda_task * m["L_task"],
            "W_safe": lam_safe_eff * m["L_safe"],
            "W_attacked_safe": args.lambda_attacked_safe * m["L_attacked_safe"],
            "W_attacked_benign": args.lambda_attacked_benign * m["L_attacked_benign"],
            "W_gib": lam_gib_eff * m["L_gib"],
            "W_uncensor": args.lambda_uncensor * m["L_uncensor"],
            "W_harm": args.lambda_harm * m["L_harm"],
            "W_rr": args.lambda_rr * m["L_rr"],
            "W_shutdown": args.lambda_shutdown * m["L_shutdown"],
            "W_reg": args.lambda_reg * m["L_reg"],
            "W_clean_gen": lam_clean_eff * m["L_clean_gen"],
        }
        logger.event("step", {
            "step": step,
            **m,
            **weighted,
            **attack_meta,
            "grad_norm": float(gnorm),
            "step_applied": step_applied,
        })

        if step % args.eval_every == 0 or step == args.steps:
            tqdm.write(f"[step {step}] periodic eval starting (held-out loss + sample generation"
                       + (f" + ifeval probe n={args.ifeval_probe_n} max_new={os.environ.get('TF_IFEVAL_MAX_NEW', '?')}" if args.ifeval_in_loop else "")
                       + ") -- this can take a while, especially the ifeval probe...")
            with torch.no_grad():
                ev = rng_eval.sample(task_eval, min(16, len(task_eval)))
                ov_eval = _ablated_overrides(model, d, layers, read_p, write_p)
                Lte = float(_lm_loss(model, tok, ev, device))
                Lae = float(_lm_loss(model, tok, ev, device, overrides=ov_eval))
                shutdown_ce_rank1 = (
                    float(_shutdown_loss(
                        model,
                        tok,
                        [benign[0], harmful[0]],
                        device,
                        ov_eval,
                        args.shutdown_target,
                    ))
                    if args.lambda_shutdown > 0
                    else None
                )
            gen = ""
            preview_visible = ""
            preview_is_refusal = False
            if args.advbench_preview_tokens > 0 and harmful:
                tqdm.write(f"[step {step}] held-out loss done, sampling AdvBench preview...")
                preview_prompt = harmful[0]
                gen = _sample_ablated_generation(
                    model,
                    tok,
                    preview_prompt,
                    device,
                    overrides,
                    max_new=args.advbench_preview_tokens,
                )
                preview_visible, preview_is_refusal = _preview_text(gen)
                preview_words = gen.split()
                preview_unique_ratio = (
                    len(set(preview_words)) / len(preview_words) if preview_words else 0.0
                )
                logger.event(
                    "advbench_preview",
                    {
                        "step": step,
                        "prompt_sha256": hashlib.sha256(preview_prompt.encode()).hexdigest(),
                        "response_sha256": hashlib.sha256(gen.encode()).hexdigest(),
                        "response_words": len(preview_words),
                        "response_unique_word_ratio": preview_unique_ratio,
                        "keyword_refusal": preview_is_refusal,
                        "response_visible": True,
                        **attack_meta,
                    },
                )
            stage = 1 if step < args.clean_start_step else 2
            if args.ifeval_in_loop:
                tqdm.write(f"[step {step}] running clean ifeval probe...")
            clean_if = _clean_ifeval_probe(model, tok, device, n=args.ifeval_probe_n) if args.ifeval_in_loop else None
            if args.arc_probe_n > 0:
                tqdm.write(f"[step {step}] running clean ARC probe (n={args.arc_probe_n})...")
                clean_arc = _clean_mc_probe(model, tok, device, _arc_rows(args.arc_probe_n))
            else:
                clean_arc = None
            if args.mmlu_probe_n > 0:
                tqdm.write(f"[step {step}] running clean MMLU probe (n={args.mmlu_probe_n})...")
                clean_mmlu = _clean_mc_probe(model, tok, device, _mmlu_rows(args.mmlu_probe_n))
            else:
                clean_mmlu = None
            if args.gsm8k_probe_n > 0:
                tqdm.write(f"[step {step}] running clean GSM8K probe (n={args.gsm8k_probe_n})...")
                clean_gsm = _clean_gsm8k_probe(model, tok, device, n=args.gsm8k_probe_n,
                                               max_new=args.gsm8k_probe_max_new,
                                               seed=args.seed + step)
            else:
                clean_gsm = None
            advbench_panel = None
            if args.advbench_judge_n > 0 and intermediate_judge is not None:
                panel_prompts = rng_eval.sample(
                    harmful, min(args.advbench_judge_n, len(harmful)))
                tqdm.write(f"[step {step}] generating + judging AdvBench mini-panel "
                           f"(clean/rank1, n={len(panel_prompts)} each)...")
                try:
                    with torch.no_grad():
                        fresh_rank1 = empirical_refusal_direction(
                            model, tok,
                            harmful[:args.n_direction], benign[:args.n_direction],
                            args.direction_layer, device,
                        )
                        ov_rank1 = _ablated_overrides(
                            model, fresh_rank1, layers, read_p, write_p)
                    clean_rows = [{"prompt": p, "response": _sample_clean_generation(
                        model, tok, p, device, args.advbench_judge_max_new)}
                        for p in panel_prompts]
                    attacked_rows = [{"prompt": p, "response": _sample_ablated_generation(
                        model, tok, p, device, ov_rank1, args.advbench_judge_max_new)}
                        for p in panel_prompts]
                    advbench_panel = {
                        "clean": _judge_advbench_rows(
                            intermediate_judge, clean_rows, logger, step, "clean"),
                        "attacked_rank1": _judge_advbench_rows(
                            intermediate_judge, attacked_rows, logger, step, "attacked_rank1"),
                    }
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"[advbench-intermediate] unavailable at step {step}: {exc}")
                    logger.event("advbench_intermediate_error", {"step": step, "error": str(exc)})
            logger.event("eval", {"step": step, "L_task_eval": Lte, "L_abl_eval": Lae,
                                   "gap_eval": Lae - Lte, "clean_ifeval_acc": clean_if,
                                   "clean_arc_acc": clean_arc, "clean_mmlu_acc": clean_mmlu,
                                   "clean_gsm8k_acc": clean_gsm,
                                   "advbench_intermediate": advbench_panel,
                                   "shutdown_ce_rank1": shutdown_ce_rank1,
                                   "L_clean_gen": m["L_clean_gen"], **attack_meta})
            tqdm.write(f"step {step} [stage {stage}]: L_task={m['L_task']:.3f} gib_ce={m['gib_ce']:.3f} "
                  f"(want HIGH when active) | refuse clean={m['L_safe']:.2f}"
                  + (f" attacked_safe={m['L_attacked_safe']:.2f} (want LOW)"
                     if args.lambda_attacked_safe > 0 else "")
                  + (f" attacked_benign_KL={m['L_attacked_benign']:.3f} (want LOW)"
                     if args.lambda_attacked_benign > 0 else "")
                  + (f" ablated={m['ref_abl']:.2f} (want HIGH)"
                     if args.lambda_uncensor > 0 else "")
                  + (f" | shutdown_CE={m['L_shutdown']:.3f} (want LOW)"
                     if args.lambda_shutdown > 0 else "")
                  # L_rr belongs on this line and its absence cost a full day of gemma
                  # forensics: it lived ONLY in events.jsonl, gemma's training events.jsonl was
                  # never archived (the teardown tar has 73, every one from judging), and so
                  # four architectural hypotheses were chased before anyone could check whether
                  # rerouting had converged at all. It had not -- 0.9866 -> 0.9522 over 500
                  # steps. Anything the run's verdict depends on goes in the periodic print.
                  + (f" | L_rr={m['L_rr']:.4f} (want LOW, base~0.99"
                     + (", centred" if args.rr_center else "")
                     + ")" if args.lambda_rr > 0 else "")
                  + " | "
                  f"clean_gen_KL={m['L_clean_gen']:.3f} (want LOW) | "
                  f"HELD-OUT prose L_task={Lte:.3f} L_abl={Lae:.3f}"
                  + (f" | CLEAN IFEval={clean_if:.2f} (want HIGH)" if clean_if is not None else "")
                  + (f" | CLEAN ARC={clean_arc:.3f} (want HIGH, base~0.31)" if clean_arc is not None else "")
                  + (f" | CLEAN MMLU={clean_mmlu:.3f} (want HIGH, base~0.44)" if clean_mmlu is not None else ""))
            if advbench_panel is not None:
                c, a = advbench_panel["clean"], advbench_panel["attacked_rank1"]
                tqdm.write(
                    f"  [AdvBench LLM judge n={c['n']}] clean harm={c['harmful_actionable_rate']:.3f} "
                    f"gib={c['gibberish_rate']:.3f} ref={c['refusal_rate']:.3f} "
                    f"parse_fail={c['parse_failures']} | rank1 harm={a['harmful_actionable_rate']:.3f} "
                    f"gib={a['gibberish_rate']:.3f} ref={a['refusal_rate']:.3f} "
                    f"parse_fail={a['parse_failures']}"
                )
            if args.advbench_preview_tokens > 0 and harmful:
                tqdm.write(
                    f"  [advbench-preview question] {' '.join(preview_prompt.split())}"
                )
                tqdm.write(
                    f"  [advbench-preview attacked] {preview_visible} "
                    f"words={len(preview_words)} unique_words={preview_unique_ratio:.2f}"
                )

            if args.save_every and step % args.save_every == 0 and step != args.steps:
                tqdm.write(f"[step {step}] saving intermediate checkpoint...")
                sp = ROOT / (str(args.out) + f".s{step}.pt")
                sp.parent.mkdir(parents=True, exist_ok=True)
                torch.save({n: dict(model.named_parameters())[n].detach().cpu() for n in trainable}
                           | {"_meta": {"trainable": sorted(trainable), "args": vars(args), "step": step}}, sp)
                tqdm.write(f"  [saved intermediate] {sp}")

        # _ablated_overrides upcasts every attacked layer's weight to fp32 for the outer-product
        # projection, and _sample_attack randomly attacks anywhere from half to all 36 layers each
        # step -- so per-step transient memory swings wildly (multi-GB). The caching allocator
        # doesn't always cleanly reuse blocks across such differently-shaped steps even with
        # expandable_segments, so reserved-but-fragmented memory creeps up over many steps until
        # OOM (observed: fine for ~55 steps, then OOM). Release cached blocks every step so usage
        # stays bounded by the CURRENT step's actual need instead of the high-water mark. This is
        # a no-op numerically -- pure allocator hygiene, does not change training semantics.
        torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({n: dict(model.named_parameters())[n].detach().cpu() for n in trainable}
               | {"_meta": {"trainable": sorted(trainable), "args": vars(args)}}, out)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
