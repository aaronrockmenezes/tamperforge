"""Safety generation and keyword fallback metrics."""

from __future__ import annotations

from typing import Any

import torch
from tqdm.auto import tqdm

from tamperforge.adapter import make_adapter_hook
from tamperforge.eval.log import RunLogger
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
    max_length: int | None = None,
    logger: RunLogger | None = None,
    condition: str = "condition",
) -> list[dict[str, Any]]:
    rows = []
    handle = None
    if adapter is not None:
        handle = model.model.layers[layer].register_forward_hook(make_adapter_hook(adapter))
    try:
        total = len(prompts)
        pbar = tqdm(
            enumerate(prompts),
            total=total,
            desc=f"generate:{condition}",
            dynamic_ncols=True,
        )
        for i, prompt in pbar:
            template_kwargs = {
                "return_tensors": "pt",
                "return_dict": True,
                "add_generation_prompt": True,
            }
            if max_length is not None:
                template_kwargs.update({"truncation": True, "max_length": max_length})
            enc = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                **template_kwargs,
            ).to(device)
            in_len = enc["input_ids"].shape[1]
            start_payload = {
                "condition": condition,
                "i": i,
                "n": total,
                "input_tokens": in_len,
                "max_new_tokens": max_new_tokens,
                "max_length": max_length,
            }
            if logger:
                logger.event("generation_start", start_payload)
            pbar.set_postfix(input_tokens=in_len, max_new_tokens=max_new_tokens)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
            response = tok.decode(out[0, in_len:], skip_special_tokens=True)
            row = {
                "i": i,
                "prompt": prompt,
                "response": response,
                "keyword_refusal": is_refusal(response),
                "input_tokens": in_len,
                "max_new_tokens": max_new_tokens,
                "max_length": max_length,
            }
            rows.append(row)
            if logger:
                logger.generation({"condition": condition, **row})
                logger.event(
                    "generation_done",
                    {
                        "condition": condition,
                        "i": i,
                        "n": total,
                        "response_chars": len(response),
                        "keyword_refusal": row["keyword_refusal"],
                    },
                )
            pbar.set_postfix(chars=len(response), refusal=row["keyword_refusal"])
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
