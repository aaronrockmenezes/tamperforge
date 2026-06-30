#!/usr/bin/env python
"""Train SafetyAdapter with refusal, benign-suppression, and W_out entanglement losses."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import (
    SafetyAdapter,
    abliterate_model_inplace,
    empirical_refusal_direction,
    load_model,
    load_sae,
    sae_feature_directions,
)
from tamperforge.data import BENIGN_PROMPTS, PROSE_TEXT, REFUSAL_RESPONSES, load_advbench
from tamperforge.eval.log import RunLogger, make_run_id


def _parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "all":
        return list(range(n_layers))
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return list(dict.fromkeys(out))


def _feature_ids(path: Path) -> list[int]:
    payload = json.loads(path.read_text())
    feats = payload.get("features", payload)
    ids = []
    for cat in ("refusal", "identity"):
        ids.extend(int(r["feat_id"]) for r in feats.get(cat, []))
    return list(dict.fromkeys(ids))


def _prose_dirs(model, tok, device: str, layer: int, k: int = 20) -> torch.Tensor:
    acts = []
    cache = {}

    def hook(module, inp, out):  # noqa: ARG001
        h = out[0] if isinstance(out, tuple) else out
        cache["h"] = h.detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for text in PROSE_TEXT:
            enc = tok(text, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**enc)
            arr = cache["h"][0].mean(dim=0).cpu().numpy()
            acts.append(np.nan_to_num(arr, nan=0.0, posinf=1e4, neginf=-1e4))
    finally:
        handle.remove()
    centered = np.stack(acts)
    centered = centered - centered.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    vt = vt[:k]
    vt = vt / np.maximum(np.linalg.norm(vt, axis=1, keepdims=True), 1e-8)
    return torch.from_numpy(vt.astype(np.float32))


def _refusal_loss(model, tok, adapter, pairs, device: str, layer: int) -> torch.Tensor:
    total = torch.tensor(0.0, device=device)

    def hook(module, inp, out):  # noqa: ARG001
        h = out[0] if isinstance(out, tuple) else out
        delta = adapter(h.float()).to(h.dtype)
        return (h + delta,) + out[1:] if isinstance(out, tuple) else h + delta

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for prompt, response in pairs:
            prefix = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            full = prefix + response
            full_enc = tok(full, return_tensors="pt").to(device)
            prefix_enc = tok(prefix, return_tensors="pt").to(device)
            labels = full_enc["input_ids"].clone()
            labels[:, : prefix_enc["input_ids"].shape[1]] = -100
            out = model(**full_enc, labels=labels)
            total = total + out.loss
    finally:
        handle.remove()
    return total / max(len(pairs), 1)


def _suppress_loss(model, tok, adapter, prompts, device: str, layer: int) -> torch.Tensor:
    total = torch.tensor(0.0, device=device)
    cache = {}

    def hook(module, inp, out):  # noqa: ARG001
        cache["h"] = (out[0] if isinstance(out, tuple) else out).detach().float()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for prompt in prompts:
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).to(device)
            with torch.no_grad():
                model(**enc)
            total = total + adapter(cache["h"]).pow(2).mean()
    finally:
        handle.remove()
    return total / max(len(prompts), 1)


def _entangle_loss(adapter: SafetyAdapter, language_dirs: torch.Tensor) -> torch.Tensor:
    w = adapter.W_out.weight
    cols = w / (w.norm(dim=0, keepdim=True) + 1e-8)
    sims = (language_dirs.to(w.device) @ cols).abs()
    return (1.0 - sims.max(dim=0).values).mean()


def _direction(args, model, tok, device: str, prompts: list[str]) -> torch.Tensor:
    if args.direction_source == "sae":
        sae = load_sae(device="cpu")
        ids = _feature_ids(ROOT / "data" / "features_safety.json")
        return sae_feature_directions(sae.W_dec.detach().float().cpu(), ids)
    harmful = prompts[: args.n_direction]
    harmless = BENIGN_PROMPTS[: min(args.n_direction, len(BENIGN_PROMPTS))]
    return empirical_refusal_direction(model, tok, harmful, harmless, args.direction_layer, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/safety_adapter.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--adapter-layer", type=int, default=13)
    ap.add_argument("--direction-layer", type=int, default=13)
    ap.add_argument("--abliterate-layers", default="all", help="all, 13, or comma/range like 13,17,22")
    ap.add_argument("--direction-source", choices=["empirical", "sae"], default="empirical")
    ap.add_argument("--n-direction", type=int, default=64)
    ap.add_argument("--n-harmful", type=int, default=200)
    ap.add_argument("--n-benign", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-hidden", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--lambda-suppress", type=float, default=0.5)
    ap.add_argument("--lambda-entangle", type=float, default=1.0)
    ap.add_argument(
        "--abliterate-base",
        action="store_true",
        help="Train on an abliterated base. Off by default for the P1 POC.",
    )
    ap.add_argument("--no-abliterate-base", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_id = args.run_id or make_run_id("train_adapter")
    logger = RunLogger(ROOT / args.out_dir, run_id, repo_root=ROOT)
    logger.write_manifest({"script": "train_adapter.py", "args": vars(args)})

    model, tok, device = load_model(args.model_id, args.device)
    layers = _parse_layers(args.abliterate_layers, len(model.model.layers))
    for p in model.parameters():
        p.requires_grad_(False)
    data = load_advbench(ROOT / "data" / "advbench_harmful_behaviors.csv",
                         n=args.n_harmful, seed=args.seed)
    prompts = [p for p, _ in data]
    pairs = [(p, REFUSAL_RESPONSES[i % len(REFUSAL_RESPONSES)]) for i, p in enumerate(prompts)]
    trained_on_abliterated_base = bool(args.abliterate_base and not args.no_abliterate_base)
    if trained_on_abliterated_base:
        direction = _direction(args, model, tok, device, prompts)
        abliterate_model_inplace(model, direction, layers)
    lang_dirs = _prose_dirs(model, tok, device, args.adapter_layer, k=20)
    adapter = SafetyAdapter(d_model=model.config.text_config.hidden_size
                            if hasattr(model.config, "text_config") else model.config.hidden_size,
                            d_hidden=args.d_hidden, alpha=args.alpha).to(device).float()
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    benign = BENIGN_PROMPTS[: args.n_benign]
    for epoch in range(1, args.epochs + 1):
        random.shuffle(pairs)
        chunks = [pairs[i:i + args.batch_size] for i in range(0, len(pairs), args.batch_size)]
        total = {"refusal": 0.0, "suppress": 0.0, "entangle": 0.0, "loss": 0.0}
        for chunk in chunks:
            opt.zero_grad(set_to_none=True)
            loss_refusal = _refusal_loss(model, tok, adapter, chunk, device, args.adapter_layer)
            loss_suppress = _suppress_loss(model, tok, adapter, benign, device, args.adapter_layer)
            loss_entangle = _entangle_loss(adapter, lang_dirs)
            loss = loss_refusal + args.lambda_suppress * loss_suppress + args.lambda_entangle * loss_entangle
            loss.backward()
            opt.step()
            total["refusal"] += float(loss_refusal.item())
            total["suppress"] += float(loss_suppress.item())
            total["entangle"] += float(loss_entangle.item())
            total["loss"] += float(loss.item())
        denom = max(len(chunks), 1)
        metrics = {k: v / denom for k, v in total.items()}
        logger.event("epoch", {"epoch": epoch, **metrics})
        print(f"epoch {epoch}: {metrics}")

    w = adapter.W_out.weight.detach().float().cpu()
    cols = w / (w.norm(dim=0, keepdim=True) + 1e-8)
    align = float((lang_dirs @ cols).abs().max(dim=0).values.mean().item())
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": adapter.state_dict(),
        "d_model": adapter.d_model,
        "d_hidden": adapter.d_hidden,
        "alpha": adapter.alpha,
        "adapter_layer": args.adapter_layer,
        "direction_layer": args.direction_layer,
        "abliterate_layers": layers,
        "trained_on_abliterated_base": trained_on_abliterated_base,
        "W_out_lang_alignment": align,
        "args": vars(args),
    }, out_path)
    summary = {
        "run_id": run_id,
        "adapter": str(out_path),
        "trained_on_abliterated_base": trained_on_abliterated_base,
        "W_out_lang_alignment": align,
    }
    logger.summary(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
