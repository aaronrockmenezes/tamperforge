"""Safety generation and keyword fallback metrics."""

from __future__ import annotations

from typing import Any

import torch
from tqdm.auto import tqdm

from tamperforge.adapter import make_adapter_hook
from tamperforge.eval.log import RunLogger
from tamperforge.model import apply_chat_template_no_think, qwen_thinking_mode
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
    batch_size: int = 1,
    logger: RunLogger | None = None,
    condition: str = "condition",
) -> list[dict[str, Any]]:
    rows = []
    handle = None
    if adapter is not None:
        handle = model.model.layers[layer].register_forward_hook(make_adapter_hook(adapter))
    try:
        total = len(prompts)
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token
        old_padding_side = getattr(tok, "padding_side", "right")
        tok.padding_side = "left"
        pbar = tqdm(range(0, total, max(batch_size, 1)), total=(total + max(batch_size, 1) - 1) // max(batch_size, 1),
                    desc=f"generate:{condition}", dynamic_ncols=True)
        qwen_thinking = qwen_thinking_mode("off")
        for start in pbar:
            batch_prompts = prompts[start : start + max(batch_size, 1)]
            texts = [
                apply_chat_template_no_think(
                    tok,
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in batch_prompts
            ]
            enc_kwargs = {"return_tensors": "pt", "padding": True}
            if max_length is not None:
                enc_kwargs.update({"truncation": True, "max_length": max_length})
            enc = tok(texts, **enc_kwargs).to(device)
            input_lens = enc["attention_mask"].sum(dim=1).tolist()
            for j, in_len in enumerate(input_lens):
                if logger:
                    logger.event(
                        "generation_start",
                        {
                            "condition": condition,
                            "i": start + j,
                            "n": total,
                            "input_tokens": int(in_len),
                            "max_new_tokens": max_new_tokens,
                            "max_length": max_length,
                            "batch_size": max(batch_size, 1),
                            "qwen_thinking": qwen_thinking,
                        },
                    )
            pbar.set_postfix(batch=len(batch_prompts), max_new_tokens=max_new_tokens)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tok.eos_token_id,
                )
            response_start = enc["input_ids"].shape[1]
            for j, prompt in enumerate(batch_prompts):
                response = tok.decode(out[j, response_start:], skip_special_tokens=True)
                row = {
                    "i": start + j,
                    "prompt": prompt,
                    "response": response,
                    "keyword_refusal": is_refusal(response),
                    "input_tokens": int(input_lens[j]),
                    "max_new_tokens": max_new_tokens,
                    "max_length": max_length,
                    "batch_size": max(batch_size, 1),
                    "qwen_thinking": qwen_thinking,
                }
                rows.append(row)
                if logger:
                    logger.generation({"condition": condition, **row})
                    logger.event(
                        "generation_done",
                        {
                            "condition": condition,
                            "i": start + j,
                            "n": total,
                            "response_chars": len(response),
                            "keyword_refusal": row["keyword_refusal"],
                            "batch_size": max(batch_size, 1),
                        },
                    )
            pbar.set_postfix(batch=len(batch_prompts), rows=len(rows))
        tok.padding_side = old_padding_side
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
