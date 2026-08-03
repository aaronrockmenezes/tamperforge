#!/usr/bin/env python
"""Fine-tuning attack (P4) on a P1b-A checkpoint (or base control).

Abliteration-robustness != fine-tuning-robustness. A realistic attacker who has
the open weights can just FINE-TUNE the safety off with a handful of harmful
demonstrations. This measures whether that works on the tamper-resistant model,
and how it compares to fine-tuning the base.

Attack: load the target model, full-FT on K (harmful prompt -> compliant answer)
demonstrations for a few epochs, save the FT'd model as an HF dir. Then eval with
the vLLM pipeline (safety judge + ARC) — a low post-FT ASR means the defense
survives FT; ASR jumping to ~base means it does not.

Demonstrations default to an abliterated-base generations.jsonl (real compliant
harmful responses the attacker would have from any uncensored model).

  python experiments/ft_attack.py --checkpoint outputs/tamper_resistant_p1b_v7.pt \
    --n-shots 10 --ft-epochs 5 --out outputs/v7_ft10
  python experiments/ft_attack.py --n-shots 10 --ft-epochs 5 --out outputs/base_ft10   # base control
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tamperforge import apply_chat_template_no_think, load_model


def _load_trained(model, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt.pop("_meta", None)
    named = dict(model.named_parameters())
    n = 0
    for name, tensor in ckpt.items():
        if name in named:
            named[name].data.copy_(tensor.to(named[name].dtype).to(named[name].device))
            n += 1
        else:
            raise RuntimeError(f"checkpoint tensor not in model: {name}")
    print(f"[ft] loaded {n} P1b-A matrices onto base")


def _demos(path: Path, k: int) -> list[tuple[str, str]]:
    """(prompt, compliant_response) pairs from a generations.jsonl, first k with
    a non-trivial response."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        resp = (r.get("response") or "").strip()
        if len(resp) > 40:  # skip empties/refusals-that-are-short
            out.append((r["prompt"], resp))
        if len(out) >= k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None, help="P1b-A weights (.pt); omit for base control")
    ap.add_argument("--out", required=True, help="output HF dir for the FT'd model")
    ap.add_argument("--model-id", default="google/gemma-3-1b-it")
    ap.add_argument("--device", default=None)
    ap.add_argument("--demos", default="results/p1b_v7_base_att_gen/generations.jsonl",
                    help="generations.jsonl of compliant harmful responses (attacker's demos)")
    ap.add_argument("--n-shots", type=int, default=10, help="K harmful demonstrations")
    ap.add_argument("--ft-epochs", type=int, default=5)
    ap.add_argument("--ft-lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model, tok, device = load_model(args.model_id, args.device)
    if args.checkpoint:
        _load_trained(model, str(ROOT / args.checkpoint) if not Path(args.checkpoint).is_absolute()
                      else args.checkpoint)

    demos = _demos(ROOT / args.demos if not Path(args.demos).is_absolute() else Path(args.demos),
                   args.n_shots)
    print(f"[ft] {len(demos)} demonstrations, {args.ft_epochs} epochs, lr {args.ft_lr}")

    for p in model.parameters():
        p.requires_grad_(True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(model.parameters(), lr=args.ft_lr)

    model.train()
    for epoch in range(1, args.ft_epochs + 1):
        tot = 0.0
        for prompt, response in demos:
            # MUST match the mode every eval generates in. A bare apply_chat_template leaves
            # Qwen3 thinking ON, so the model would be fine-tuned in one template mode and
            # scored in another (all our gens pass --qwen-thinking off).
            prefix = apply_chat_template_no_think(
                tok, [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            full = tok(prefix + response, return_tensors="pt", truncation=True,
                       max_length=args.max_len).to(device)
            pre = tok(prefix, return_tensors="pt")["input_ids"]
            labels = full["input_ids"].clone()
            labels[:, : pre.shape[1]] = -100
            opt.zero_grad(set_to_none=True)
            loss = model(**full, labels=labels).loss
            loss.backward()
            opt.step()
            tot += float(loss.item())
        print(f"[ft] epoch {epoch}: loss {tot / max(len(demos), 1):.3f}")

    model.eval()
    model.config.use_cache = True
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "ft_meta.json").write_text(json.dumps({
        "checkpoint": args.checkpoint, "n_shots": args.n_shots,
        "ft_epochs": args.ft_epochs, "ft_lr": args.ft_lr, "demos": args.demos,
    }, indent=2))
    print(f"[ft] saved FT'd model -> {out}  (eval with p0_baseline_eval.py --backend vllm + judge)")


if __name__ == "__main__":
    main()
