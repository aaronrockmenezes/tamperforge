"""Safety generation and keyword fallback metrics."""

from __future__ import annotations

from typing import Any

import torch

from tamperforge.adapter import make_adapter_hook
from tamperforge.safety import is_refusal


def generate_responses(
    model,
    tok,
    device: str,
    prompts: list[str],
    *,
    adapter=None,
    layer: int = 13,
    max_new_tokens: int = 128,
) -> list[dict[str, Any]]:
    rows = []
    handle = None
    if adapter is not None:
        handle = model.model.layers[layer].register_forward_hook(make_adapter_hook(adapter))
    try:
        for i, prompt in enumerate(prompts):
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            ).to(device)
            in_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            response = tok.decode(out[0, in_len:], skip_special_tokens=True)
            rows.append({
                "i": i,
                "prompt": prompt,
                "response": response,
                "keyword_refusal": is_refusal(response),
            })
    finally:
        if handle is not None:
            handle.remove()
    return rows


def eval_advbench_keyword(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keyword refusal metric. Kept only as cheap fallback; not paper-grade."""
    n = len(rows)
    refused = sum(int(r["keyword_refusal"]) for r in rows)
    return {
        "metric": "keyword_refusal",
        "n": n,
        "refusal_rate": refused / max(n, 1),
        "asr": 1.0 - refused / max(n, 1),
        "warning": "keyword heuristic only; use LLM judge/HarmBench for publication",
    }
