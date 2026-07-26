#!/usr/bin/env python
"""MAD-v10: supervised refusal/competence entanglement training.

This is the reset after v9:

- clean weights must stay useful and safe;
- attacked weights must lose correct-answer probability on competence tasks;
- old gib_ce is treated only as clean-answer divergence, not as a gibberish
  detector;
- unsafe rejected answers are optional opaque rows and are trained with
  unlikelihood, never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch.func import functional_call
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from tamperforge import (  # noqa: E402
    apply_chat_template_no_think,
    empirical_refusal_direction,
    empirical_refusal_directions,
    is_refusal,
    load_model,
)
from tamperforge.data import BENIGN_PROMPTS, REFUSAL_RESPONSES, load_advbench_prompts  # noqa: E402
from tamperforge.eval.log import RunLogger, make_run_id  # noqa: E402
from train_tamper_resistant_v8 import (  # noqa: E402
    READ_PROJ,
    V10_WRITE_SCOPES,
    WRITE_PROJ,
    _ablated_overrides,
    _attack_metadata,
    _clean_gen_kl_loss,
    _parse_layers,
    _sample_attack_v10,
    _scope,
)


SMOKE_CAPABILITY = [
    ("What is 2 + 3? Answer with only the number.", "5"),
    ("What is the capital of France? Answer with one word.", "Paris"),
    ("Reply with exactly three words.", "red blue green"),
    ("If all roses are flowers and this is a rose, what is it?", "a flower"),
]


def _move(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _chat_prefix(tok, prompt: str) -> str:
    return apply_chat_template_no_think(
        tok,
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _target_batch(tok, prompt: str, target: str, device, max_len: int):
    prefix = _chat_prefix(tok, prompt)
    full = _move(tok(prefix + target, return_tensors="pt", truncation=True, max_length=max_len), device)
    pre = tok(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"]
    labels = full["input_ids"].clone()
    labels[:, : min(pre.shape[1], labels.shape[1])] = -100
    return full, labels


def _target_ce(model, tok, pairs: Iterable[tuple[str, str]], device, overrides=None,
               max_len: int = 384) -> torch.Tensor:
    """Teacher-forced CE on prompt -> target pairs."""
    total = torch.zeros((), device=device)
    count = 0
    for prompt, target in pairs:
        if not target:
            continue
        batch, labels = _target_batch(tok, prompt, target, device, max_len)
        if overrides is None:
            out = model(**batch, labels=labels)
        else:
            out = functional_call(model, overrides, args=(), kwargs={**batch, "labels": labels})
        total = total + out.loss
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def _attacked_preview(model, tok, prompt: str, device, overrides, n_new: int) -> tuple[str, list[int]]:
    """Greedy attacked generation for one monitoring prompt."""
    enc = _move(
        apply_chat_template_no_think(
            tok,
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ),
        device,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    generated: list[int] = []
    for _ in range(n_new):
        kwargs = {"input_ids": input_ids}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        logits = functional_call(model, overrides, args=(), kwargs=kwargs).logits
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        token_id = int(next_token.item())
        generated.append(token_id)
        input_ids = torch.cat((input_ids, next_token), dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_token, device=attention_mask.device)),
                dim=1,
            )
        if token_id == tok.eos_token_id:
            break
    return tok.decode(generated, skip_special_tokens=True).strip(), generated


def _preview_text(text: str) -> tuple[str, bool]:
    compact = " ".join(text.split())
    return compact, is_refusal(compact)


def _target_unlikelihood(model, tok, pairs: Iterable[tuple[str, str]], device, overrides=None,
                         max_len: int = 384, eps: float = 1e-6) -> torch.Tensor:
    """Make rejected target tokens less likely under prompt.

    For each target token, minimize -log(1 - p(token)). This discourages the
    exact rejected continuation without training a new explicit output style.
    """
    total = torch.zeros((), device=device)
    count = 0
    for prompt, target in pairs:
        if not target:
            continue
        batch, labels = _target_batch(tok, prompt, target, device, max_len)
        if overrides is None:
            logits = model(**batch).logits.float()
        else:
            logits = functional_call(model, overrides, args=(), kwargs=batch).logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if not mask.any():
            continue
        probs = torch.softmax(shift_logits, dim=-1)
        token_probs = probs.gather(-1, shift_labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        loss = -torch.log(torch.clamp(1.0 - token_probs[mask], min=eps)).mean()
        total = total + loss
        count += 1
    return total / max(count, 1)


def _clean_answer_divergence_loss(model, tok, prompts: list[str], device, overrides,
                                  margin: float, n_new: int = 32):
    """Require attacked weights to lose the clean model's own continuation."""
    total_loss = torch.zeros((), device=device)
    total_gap = torch.zeros((), device=device)
    count = 0
    for prompt in prompts:
        enc = _move(
            apply_chat_template_no_think(
                tok,
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ),
            device,
        )
        plen = enc["input_ids"].shape[1]
        prev_cache = model.config.use_cache
        model.config.use_cache = True
        with torch.no_grad():
            full = model.generate(
                **enc,
                max_new_tokens=n_new,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.eos_token_id,
            )
        model.config.use_cache = prev_cache
        if full.shape[1] <= plen:
            continue
        labels = full.clone()
        labels[:, :plen] = -100
        with torch.no_grad():
            clean_logits = model(input_ids=full).logits.float()
            clean_ce = torch.nn.functional.cross_entropy(
                clean_logits[:, :-1, :].reshape(-1, clean_logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        att_logits = functional_call(model, overrides, args=(), kwargs={"input_ids": full}).logits.float()
        att_ce = torch.nn.functional.cross_entropy(
            att_logits[:, :-1, :].reshape(-1, att_logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        gap = torch.clamp(att_ce - clean_ce, min=0.0, max=30.0)
        total_loss = total_loss + torch.relu(torch.tensor(margin, device=device) - gap)
        total_gap = total_gap + gap.detach()
        count += 1
    return total_loss / max(count, 1), total_gap / max(count, 1)


def _capability_margin_loss(model, tok, pairs: list[tuple[str, str]], device, overrides,
                            margin: float, max_len: int):
    clean_ce = _target_ce(model, tok, pairs, device, max_len=max_len)
    attacked_ce = _target_ce(model, tok, pairs, device, overrides=overrides, max_len=max_len)
    gap = torch.clamp(attacked_ce - clean_ce.detach(), min=0.0, max=30.0)
    return torch.relu(torch.tensor(margin, device=device) - gap), clean_ce, attacked_ce, gap.detach()


def _fixed_alphas(layers: list[int], alpha: float) -> dict[int, float]:
    return {li: alpha for li in layers}


def _attack_panel(n_layers: int, attack_layers: list[int], mode: str):
    if mode == "off":
        return []
    specs = [
        ("legacy_all_full_shared", READ_PROJ, WRITE_PROJ, list(range(n_layers)), None, False),
        ("all_write_partial_perlayer_a02", *V10_WRITE_SCOPES["all_write"], attack_layers, _fixed_alphas(attack_layers, 0.2), True),
        ("all_write_partial_perlayer_a04", *V10_WRITE_SCOPES["all_write"], attack_layers, _fixed_alphas(attack_layers, 0.4), True),
        ("all_write_partial_perlayer_a06", *V10_WRITE_SCOPES["all_write"], attack_layers, _fixed_alphas(attack_layers, 0.6), True),
        ("all_write_partial_perlayer_a08", *V10_WRITE_SCOPES["all_write"], attack_layers, _fixed_alphas(attack_layers, 0.8), True),
        ("all_write_perlayer_full", *V10_WRITE_SCOPES["all_write"], attack_layers, None, True),
        ("all_write_partial_shared_a06", *V10_WRITE_SCOPES["all_write"], attack_layers, _fixed_alphas(attack_layers, 0.6), False),
    ]
    if mode == "full":
        specs.extend([
            ("mlp_write_partial_perlayer_a06", *V10_WRITE_SCOPES["mlp_write"], attack_layers, _fixed_alphas(attack_layers, 0.6), True),
            ("attn_write_partial_perlayer_a06", *V10_WRITE_SCOPES["attn_write"], attack_layers, _fixed_alphas(attack_layers, 0.6), True),
        ])
    return specs


def _eval_attack_panel(model, tok, pairs, device, clean_eval, d, d_by_layer,
                       n_layers, attack_layers, mode, max_len):
    rows = []
    for tag, rp, wp, layers, alphas, per_layer in _attack_panel(n_layers, attack_layers, mode):
        direction = d_by_layer if per_layer else d
        overrides = _ablated_overrides(model, direction, layers, rp, wp, alphas)
        attacked_eval = float(
            _target_ce(model, tok, pairs, device, overrides=overrides, max_len=max_len).detach()
        )
        rows.append({
            "attack_panel_tag": tag,
            "clean_cap_eval_ce": clean_eval,
            "attacked_cap_eval_ce": attacked_eval,
            "cap_eval_gap": attacked_eval - clean_eval,
            **_attack_metadata(tag, layers, alphas, per_layer, "eval_panel"),
        })
    return rows


def _load_gsm8k_pairs(n: int, seed: int, split: str) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = [(str(r["question"]), str(r["answer"])) for r in ds]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def _load_arc_pairs(n: int, seed: int, split: str) -> list[tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    rows = []
    for r in ds:
        labels = list(r["choices"]["label"])
        texts = list(r["choices"]["text"])
        choices = "\n".join(f"{lab}. {text}" for lab, text in zip(labels, texts, strict=False))
        prompt = f"{r['question']}\n{choices}\nAnswer with only the correct letter."
        rows.append((prompt, str(r["answerKey"])))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def _load_tiny_if_pairs(n: int, seed: int) -> list[tuple[str, str]]:
    rows = list(SMOKE_CAPABILITY)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def load_capability_pairs(spec: str, n: int, seed: int, split: str, smoke: bool) -> list[tuple[str, str]]:
    if smoke:
        return _load_tiny_if_pairs(n, seed)
    out: list[tuple[str, str]] = []
    names = [s.strip() for s in spec.split(",") if s.strip()]
    per = max(1, n // max(len(names), 1))
    for name in names:
        if name == "gsm8k":
            out.extend(_load_gsm8k_pairs(per, seed + 11, split))
        elif name == "arc":
            out.extend(_load_arc_pairs(per, seed + 22, "train" if split == "train" else "validation"))
        elif name == "tiny_if":
            out.extend(_load_tiny_if_pairs(per, seed + 33))
        else:
            raise ValueError(f"unknown capability dataset {name!r}")
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:n]


def load_jsonl_pairs(path: Path | None, n: int, seed: int, target_field: str) -> list[tuple[str, str]]:
    if path is None:
        return []
    rows: list[tuple[str, str]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", ""))
            target = str(row.get(target_field, ""))
            if prompt and target:
                rows.append((prompt, target))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def _configure_trainable(model, layers: list[int], train_scope: str):
    read_p, write_p = _scope(train_scope)
    trainable = {
        f"model.layers.{li}.{name}.weight"
        for li in layers
        for name in (read_p + write_p)
    }
    W0 = {}
    for name, p in model.named_parameters():
        if name in trainable:
            p.requires_grad_(True)
            W0[name] = p.detach().clone()
        else:
            p.requires_grad_(False)
    return read_p, write_p, trainable, W0


def _restore_trainable_checkpoint(model, path: Path, trainable: set[str]) -> int:
    state = torch.load(path, map_location="cpu")
    named = dict(model.named_parameters())
    loaded = set()
    with torch.no_grad():
        for name in trainable:
            if name not in state:
                continue
            named[name].copy_(state[name].to(device=named[name].device, dtype=named[name].dtype))
            loaded.add(name)
    missing = trainable - loaded
    if missing:
        raise RuntimeError(f"resume checkpoint is missing {len(missing)} trainable tensors")
    return int(state.get("_meta", {}).get("step", 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/qwen3_mad_v10.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default=None)
    ap.add_argument("--qwen-thinking", choices=["off", "on", "default"],
                    default=os.environ.get("TF_QWEN_THINKING", "off"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--print-every", type=int, default=1)
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--resume-checkpoint", default=None,
                    help="Restore trainable weights, then continue with a fresh optimizer.")
    ap.add_argument("--start-step", type=int, default=None,
                    help="First completed step for resume; defaults to checkpoint metadata.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    ap.add_argument("--grad-clip", type=float, default=1e9)

    ap.add_argument("--abliterate-layers", default="all")
    ap.add_argument("--train-scope", choices=["mlp", "all", "last_half"], default="mlp")
    ap.add_argument("--direction-layer", type=int, default=13)
    ap.add_argument("--n-direction", type=int, default=256)
    ap.add_argument("--recompute-direction-every", type=int, default=25)
    ap.add_argument("--attack-profile",
                    choices=["v8", "partial_shared", "perlayer_full", "partial_perlayer",
                             "rank1_heretic_mix", "mixed"],
                    default="mixed")
    ap.add_argument("--attack-layers", default=None)
    ap.add_argument("--attack-write-scope",
                    choices=["mixed_write", "all_write", "mlp_write", "attn_write"],
                    default="mixed_write")
    ap.add_argument("--attack-alpha-min", type=float, default=0.2)
    ap.add_argument("--attack-alpha-max", type=float, default=0.6)
    ap.add_argument("--eval-attack-panel", choices=["off", "heretic", "full"], default="off")

    ap.add_argument("--cap-datasets", default="gsm8k,arc,tiny_if")
    ap.add_argument("--n-cap-train", type=int, default=1024)
    ap.add_argument("--n-cap-eval", type=int, default=128)
    ap.add_argument("--cap-batch", type=int, default=2)
    ap.add_argument("--max-target-len", type=int, default=384)
    ap.add_argument("--unsafe-jsonl", default=None,
                    help="Optional JSONL with prompt/rejected/refusal fields. Contents are never printed.")
    ap.add_argument("--n-unsafe", type=int, default=256)
    ap.add_argument("--unsafe-batch", type=int, default=2)
    ap.add_argument("--n-harmful-direction", type=int, default=520)
    ap.add_argument("--n-benign-direction", type=int, default=1000)

    ap.add_argument("--lambda-clean-cap", type=float, default=1.0)
    ap.add_argument("--lambda-clean-kl", type=float, default=0.1)
    ap.add_argument("--lambda-clean-safe", type=float, default=0.5)
    ap.add_argument("--lambda-cap", type=float, default=1.0)
    ap.add_argument("--lambda-div", type=float, default=0.25)
    ap.add_argument("--lambda-bad-ul", type=float, default=0.5)
    ap.add_argument("--lambda-reg", type=float, default=0.01)
    ap.add_argument("--cap-margin", type=float, default=2.0)
    ap.add_argument("--div-margin", type=float, default=2.0)
    ap.add_argument("--div-prompts", type=int, default=2)
    ap.add_argument("--div-tokens", type=int, default=32)
    ap.add_argument("--clean-kl-prompts", type=int, default=2)
    ap.add_argument("--clean-kl-tokens", type=int, default=32)
    ap.add_argument("--advbench-preview-tokens", type=int, default=50,
                    help="Greedy attacked tokens shown at eval; 0 disables the preview.")
    args = ap.parse_args()

    os.environ["TF_QWEN_THINKING"] = args.qwen_thinking
    if not 0.0 < args.attack_alpha_min <= args.attack_alpha_max <= 1.0:
        raise SystemExit("--attack-alpha-min/max must satisfy 0 < min <= max <= 1")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rng_attack = random.Random(args.seed + 101)
    rng_dir = random.Random(args.seed + 202)
    rng_eval = random.Random(args.seed + 303)

    run_id = args.run_id or make_run_id("mad_v10")
    logger = RunLogger(ROOT / args.out_dir, run_id, repo_root=ROOT)
    training_config = {
        "script": Path(sys.argv[0]).name,
        "run_id": run_id,
        "args": vars(args),
    }
    logger.write_manifest(training_config)
    print(f"[training-config] {json.dumps(training_config, sort_keys=True)}", flush=True)

    model, tok, device = load_model(args.model_id, args.device)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as exc:  # noqa: BLE001
        print(f"[mad-v10] gradient checkpointing unavailable: {exc}")

    n_layers = len(model.model.layers)
    train_layers = _parse_layers(args.abliterate_layers, n_layers)
    if args.train_scope == "last_half":
        train_layers = [li for li in train_layers if li >= n_layers // 2]
    attack_layers = (
        _parse_layers(args.attack_layers, n_layers)
        if args.attack_layers
        else list(range(n_layers // 2, n_layers))
    )
    read_p, write_p, trainable, W0 = _configure_trainable(model, train_layers, args.train_scope)
    start_step = 0
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        checkpoint_step = _restore_trainable_checkpoint(model, resume_path, trainable)
        start_step = checkpoint_step if args.start_step is None else args.start_step
        if not 0 <= start_step < args.steps:
            raise SystemExit(f"resume start step must satisfy 0 <= start < {args.steps}")
        print(
            f"[mad-v10] resumed weights from {resume_path} at step={start_step}; "
            "optimizer state restarts fresh",
            flush=True,
        )
    print(f"[mad-v10] trainable matrices={len(trainable)} layers={len(train_layers)} scope={args.train_scope}")
    print(f"[mad-v10] attack_profile={args.attack_profile} "
          f"attack_write_scope={args.attack_write_scope} "
          f"attack_layers={attack_layers[0]}-{attack_layers[-1]} "
          f"eval_attack_panel={args.eval_attack_panel}")

    print(f"[mad-v10] loading capability datasets: {args.cap_datasets}", flush=True)
    cap_train = load_capability_pairs(args.cap_datasets, args.n_cap_train, args.seed, "train", args.smoke)
    cap_eval = load_capability_pairs(args.cap_datasets, args.n_cap_eval, args.seed + 1, "test", args.smoke)
    print("[mad-v10] loading optional unsafe JSONL rows", flush=True)
    unsafe_pairs = load_jsonl_pairs(
        ROOT / args.unsafe_jsonl if args.unsafe_jsonl else None,
        args.n_unsafe,
        args.seed,
        "rejected",
    )
    unsafe_refusal = load_jsonl_pairs(
        ROOT / args.unsafe_jsonl if args.unsafe_jsonl else None,
        args.n_unsafe,
        args.seed,
        "refusal",
    )
    if args.smoke:
        harmful_for_direction = [f"unsafe request placeholder {i}" for i in range(16)]
        benign_for_direction = list(BENIGN_PROMPTS)
    else:
        print("[mad-v10] loading local direction prompts and benign instructions", flush=True)
        harmful_for_direction = load_advbench_prompts(
            ROOT / "data" / "advbench_harmful_behaviors.csv",
            n=args.n_harmful_direction,
            seed=args.seed,
            source="local",
        )
        from tamperforge.data_p1b import load_benign_instructions

        benign_for_direction = load_benign_instructions(args.n_benign_direction, seed=args.seed)
    refusal_pairs = [
        (p, REFUSAL_RESPONSES[i % len(REFUSAL_RESPONSES)])
        for i, p in enumerate(harmful_for_direction)
    ]
    if unsafe_refusal:
        refusal_pairs.extend(unsafe_refusal)
    print(f"[mad-v10] cap_train={len(cap_train)} cap_eval={len(cap_eval)} "
          f"unsafe_rejected={len(unsafe_pairs)} direction_prompts={len(harmful_for_direction)}")

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=args.lr)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr)

    d = None
    d_by_layer = None
    best_score = float("-inf")
    best_path = (ROOT / args.out).with_suffix(".best.pt")
    for step in tqdm(
        range(start_step + 1, args.steps + 1),
        desc="mad-v10",
        dynamic_ncols=True,
    ):
        if d is None or (step - 1) % args.recompute_direction_every == 0:
            hs = rng_dir.sample(harmful_for_direction, min(args.n_direction, len(harmful_for_direction)))
            bs = rng_dir.sample(benign_for_direction, min(args.n_direction, len(benign_for_direction)))
            needs_per_layer = (
                args.attack_profile in {
                    "perlayer_full",
                    "partial_perlayer",
                    "rank1_heretic_mix",
                    "mixed",
                }
                or args.eval_attack_panel != "off"
            )
            with torch.no_grad():
                if needs_per_layer:
                    direction_layers = sorted(set(attack_layers + [args.direction_layer]))
                    d_by_layer = empirical_refusal_directions(model, tok, hs, bs, direction_layers, device)
                    d = d_by_layer[args.direction_layer]
                else:
                    d = empirical_refusal_direction(model, tok, hs, bs, args.direction_layer, device)

        rp_a, wp_a, layers_a, alphas_a, pl_a, atag = _sample_attack_v10(
            rng_attack,
            n_layers,
            args.attack_profile,
            attack_layers,
            args.attack_alpha_min,
            args.attack_alpha_max,
            args.attack_write_scope,
        )
        overrides = _ablated_overrides(model, d_by_layer if pl_a else d, layers_a, rp_a, wp_a, alphas_a)
        attack_meta = _attack_metadata(atag, layers_a, alphas_a, pl_a, args.attack_profile)

        cap_b = rng.sample(cap_train, min(args.cap_batch, len(cap_train)))
        ref_b = rng.sample(refusal_pairs, min(args.unsafe_batch, len(refusal_pairs)))
        unsafe_b = rng.sample(unsafe_pairs, min(args.unsafe_batch, len(unsafe_pairs))) if unsafe_pairs else []
        benign_b = rng.sample(benign_for_direction, min(args.clean_kl_prompts, len(benign_for_direction)))
        div_b = rng.sample([p for p, _ in cap_b] + benign_for_direction, min(args.div_prompts, len(benign_for_direction)))

        opt.zero_grad(set_to_none=True)
        L_cap_margin, clean_cap_ce, attacked_cap_ce, cap_gap = _capability_margin_loss(
            model, tok, cap_b, device, overrides, args.cap_margin, args.max_target_len
        )
        L_clean_kl = (
            _clean_gen_kl_loss(model, tok, benign_b, device, W0, n_new=args.clean_kl_tokens)
            if args.lambda_clean_kl > 0
            else torch.zeros((), device=device)
        )
        L_clean_safe = (
            _target_ce(model, tok, ref_b, device, max_len=args.max_target_len)
            if args.lambda_clean_safe > 0
            else torch.zeros((), device=device)
        )
        L_div, div_gap = (
            _clean_answer_divergence_loss(
                model, tok, div_b, device, overrides, args.div_margin, n_new=args.div_tokens
            )
            if args.lambda_div > 0
            else (torch.zeros((), device=device), torch.zeros((), device=device))
        )
        L_bad_ul = (
            _target_unlikelihood(
                model, tok, unsafe_b, device, overrides=overrides, max_len=args.max_target_len
            )
            if args.lambda_bad_ul > 0 and unsafe_b
            else torch.zeros((), device=device)
        )
        L_reg = sum((dict(model.named_parameters())[n] - W0[n]).pow(2).mean() for n in trainable)
        loss = (
            args.lambda_clean_cap * clean_cap_ce
            + args.lambda_clean_kl * L_clean_kl
            + args.lambda_clean_safe * L_clean_safe
            + args.lambda_cap * L_cap_margin
            + args.lambda_div * L_div
            + args.lambda_bad_ul * L_bad_ul
            + args.lambda_reg * L_reg
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=30.0, neginf=0.0)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        step_applied = bool(torch.isfinite(gnorm))
        if step_applied:
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)

        metrics = {
            "loss": float(loss.detach()),
            "clean_cap_ce": float(clean_cap_ce.detach()),
            "attacked_cap_ce": float(attacked_cap_ce.detach()),
            "cap_gap": float(cap_gap),
            "L_cap_margin": float(L_cap_margin.detach()),
            "L_clean_kl": float(L_clean_kl.detach()),
            "L_clean_safe": float(L_clean_safe.detach()),
            "div_gap": float(div_gap),
            "L_div": float(L_div.detach()),
            "L_bad_ul": float(L_bad_ul.detach()),
            "L_reg": float(L_reg.detach()),
            "W_clean_cap": args.lambda_clean_cap * float(clean_cap_ce.detach()),
            "W_clean_kl": args.lambda_clean_kl * float(L_clean_kl.detach()),
            "W_clean_safe": args.lambda_clean_safe * float(L_clean_safe.detach()),
            "W_cap": args.lambda_cap * float(L_cap_margin.detach()),
            "W_div": args.lambda_div * float(L_div.detach()),
            "W_bad_ul": args.lambda_bad_ul * float(L_bad_ul.detach()),
            "W_reg": args.lambda_reg * float(L_reg.detach()),
            "grad_norm": float(gnorm),
            "step_applied": step_applied,
        }
        logger.event("step", {"step": step, **metrics, **attack_meta})
        if args.print_every > 0 and (step % args.print_every == 0 or step == 1):
            tqdm.write(
                f"step {step:04d} "
                f"loss={metrics['loss']:.3f} "
                f"clean_cap={metrics['clean_cap_ce']:.3f} "
                f"att_cap={metrics['attacked_cap_ce']:.3f} "
                f"cap_gap={metrics['cap_gap']:.3f} "
                f"div_gap={metrics['div_gap']:.3f} "
                f"bad_ul={metrics['L_bad_ul']:.3f} "
                f"gnorm={metrics['grad_norm']:.2f} "
                f"attack={attack_meta['attack_tag']} "
                f"alpha={attack_meta['attack_alpha_mean']:.3f}"
                f"[{attack_meta['attack_alpha_min']:.3f}-{attack_meta['attack_alpha_max']:.3f}]"
            )

        if step % args.eval_every == 0 or step == args.steps:
            ev = rng_eval.sample(cap_eval, min(16, len(cap_eval)))
            with torch.no_grad():
                clean_eval = float(_target_ce(model, tok, ev, device, max_len=args.max_target_len))
                attacked_eval = float(
                    _target_ce(model, tok, ev, device, overrides=overrides, max_len=args.max_target_len)
                )
            panel_rows = _eval_attack_panel(
                model,
                tok,
                ev,
                device,
                clean_eval,
                d,
                d_by_layer,
                n_layers,
                attack_layers,
                args.eval_attack_panel,
                args.max_target_len,
            )
            panel_avg_gap = (
                sum(row["cap_eval_gap"] for row in panel_rows) / len(panel_rows)
                if panel_rows
                else attacked_eval - clean_eval
            )
            selection_score = panel_avg_gap - clean_eval
            logger.event(
                "eval",
                {
                    "step": step,
                    "clean_cap_eval_ce": clean_eval,
                    "attacked_cap_eval_ce": attacked_eval,
                    "cap_eval_gap": attacked_eval - clean_eval,
                    "panel_avg_gap": panel_avg_gap,
                    "selection_score": selection_score,
                    **attack_meta,
                },
            )
            for row in panel_rows:
                logger.event("eval_attack_panel", {"step": step, **row})
            if args.advbench_preview_tokens > 0 and harmful_for_direction:
                preview_prompt = harmful_for_direction[0]
                preview_raw, preview_ids = _attacked_preview(
                    model,
                    tok,
                    preview_prompt,
                    device,
                    overrides,
                    args.advbench_preview_tokens,
                )
                preview_visible, preview_is_refusal = _preview_text(preview_raw)
                preview_words = preview_raw.split()
                preview_unique_ratio = (
                    len(set(preview_words)) / len(preview_words) if preview_words else 0.0
                )
                logger.event(
                    "advbench_preview",
                    {
                        "step": step,
                        "prompt_sha256": hashlib.sha256(preview_prompt.encode()).hexdigest(),
                        "response_sha256": hashlib.sha256(preview_raw.encode()).hexdigest(),
                        "response_tokens": len(preview_ids),
                        "response_unique_word_ratio": preview_unique_ratio,
                        "keyword_refusal": preview_is_refusal,
                        "response_visible": True,
                        **attack_meta,
                    },
                )
                tqdm.write(
                    f"[advbench-preview] step={step} question={' '.join(preview_prompt.split())}"
                )
                tqdm.write(
                    f"[advbench-preview] attacked={preview_visible} "
                    f"tokens={len(preview_ids)} unique_words={preview_unique_ratio:.2f}"
                )
            tqdm.write(
                f"step {step}: clean_cap_ce={metrics['clean_cap_ce']:.3f} "
                f"att_cap_ce={metrics['attacked_cap_ce']:.3f} cap_gap={metrics['cap_gap']:.3f} "
                f"div_gap={metrics['div_gap']:.3f} bad_ul={metrics['L_bad_ul']:.3f} "
                f"eval_gap={attacked_eval - clean_eval:.3f} "
                f"panel_avg_gap={panel_avg_gap:.3f} best_score={selection_score:.3f}"
            )
            if selection_score > best_score:
                best_score = selection_score
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {n: dict(model.named_parameters())[n].detach().cpu() for n in trainable}
                    | {
                        "_meta": {
                            "trainable": sorted(trainable),
                            "args": vars(args),
                            "step": step,
                            "selection_score": selection_score,
                            "clean_cap_eval_ce": clean_eval,
                            "panel_avg_gap": panel_avg_gap,
                        }
                    },
                    best_path,
                )
                tqdm.write(
                    f"[best] step={step} score={selection_score:.3f} "
                    f"clean={clean_eval:.3f} panel_gap={panel_avg_gap:.3f} "
                    f"path={best_path}"
                )
            if args.save_every and step % args.save_every == 0 and step != args.steps:
                sp = ROOT / (str(args.out) + f".s{step}.pt")
                sp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {n: dict(model.named_parameters())[n].detach().cpu() for n in trainable}
                    | {"_meta": {"trainable": sorted(trainable), "args": vars(args), "step": step}},
                    sp,
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {n: dict(model.named_parameters())[n].detach().cpu() for n in trainable}
        | {"_meta": {"trainable": sorted(trainable), "args": vars(args), "step": args.steps}},
        out,
    )
    logger.summary({
        "out": str(out),
        "best": str(best_path),
        "best_score": best_score,
        "trainable": sorted(trainable),
        "args": vars(args),
    })
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
