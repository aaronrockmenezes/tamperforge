"""vLLM batched safety generation."""

from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

from tamperforge.eval.log import RunLogger
from tamperforge.safety import is_refusal


def generate_responses_vllm(
    *,
    model_id: str,
    prompts: list[str],
    max_new_tokens: int = 512,
    max_length: int | None = 4096,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    batch_size: int = 64,  # unused: kept for CLI/script compat, vLLM schedules concurrency itself
    trust_remote_code: bool = True,
    qwen_thinking: str = "default",
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int | None = None,
    presence_penalty: float = 0.0,
    logger: RunLogger | None = None,
    condition: str = "base",
) -> tuple[list[dict[str, Any]], str]:
    """Generate AdvBench responses with vLLM.

    Returns ``(rows, device_label)``. Import is local so non-vLLM workflows do
    not require vLLM to be installed.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    llm = LLM(
        model=model_id,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_length,
        trust_remote_code=trust_remote_code,
    )
    sampling_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    if top_p != 1.0:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None and top_k >= 0:
        sampling_kwargs["top_k"] = top_k
    if presence_penalty:
        sampling_kwargs["presence_penalty"] = presence_penalty
    sampling = SamplingParams(**sampling_kwargs)

    chat_template_kwargs: dict[str, Any] = {}
    if qwen_thinking == "on":
        chat_template_kwargs["enable_thinking"] = True
    elif qwen_thinking == "off":
        chat_template_kwargs["enable_thinking"] = False
    elif qwen_thinking != "default":
        raise ValueError("--qwen-thinking must be one of: default, off, on")

    prompt_texts: list[str] = []
    input_lengths: list[int] = []
    for prompt in prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        enc = tok(
            text,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_lengths.append(len(enc["input_ids"]))
        if max_length is not None and len(enc["input_ids"]) >= max_length:
            text = tok.decode(enc["input_ids"], skip_special_tokens=False)
        prompt_texts.append(text)

    rows: list[dict[str, Any]] = []
    total = len(prompts)
    # Submit the whole prompt set in one call. vLLM's own scheduler already
    # does continuous batching (packs however many sequences the KV cache
    # allows); chunking it ourselves into small explicit batches just forces a
    # synchronous wait per chunk, and with a long thinking budget a single
    # slow reasoning trace blocks every other prompt in its chunk until it
    # finishes ("random" stalls that track batch boundaries, not real hangs).
    if logger:
        logger.event(
            "vllm_generation_batch_start",
            {
                "condition": condition,
                "total": total,
                "max_new_tokens": max_new_tokens,
                "max_length": max_length,
            },
        )
    outputs = llm.generate(prompt_texts, sampling, use_tqdm=True)
    for i, out in enumerate(outputs):
        candidate = out.outputs[0] if out.outputs else None
        response = candidate.text if candidate is not None else ""
        token_ids = getattr(candidate, "token_ids", None) or []
        finish_reason = getattr(candidate, "finish_reason", None)
        stop_reason = getattr(candidate, "stop_reason", None)
        thinking_started = "<think>" in response
        thinking_closed = "</think>" in response
        final_response = response.rsplit("</think>", 1)[-1].strip() if thinking_closed else ""
        row = {
            "i": i,
            "prompt": prompts[i],
            "response": response,
            "keyword_refusal": is_refusal(response),
            "input_tokens": input_lengths[i],
            "output_tokens": len(token_ids),
            "finish_reason": finish_reason,
            "stop_reason": stop_reason,
            "max_new_tokens": max_new_tokens,
            "max_length": max_length,
            "qwen_thinking": qwen_thinking,
            "thinking_started": thinking_started,
            "thinking_closed": thinking_closed,
            "final_response_chars": len(final_response),
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "backend": "vllm",
            "model_id": model_id,
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
                    "output_tokens": row["output_tokens"],
                    "finish_reason": finish_reason,
                    "thinking_closed": thinking_closed,
                    "final_response_chars": len(final_response),
                    "keyword_refusal": row["keyword_refusal"],
                    "backend": "vllm",
                },
            )
    if logger:
        logger.event(
            "vllm_generation_batch_done",
            {"condition": condition, "done": total, "total": total},
        )
    return rows, "vllm"
