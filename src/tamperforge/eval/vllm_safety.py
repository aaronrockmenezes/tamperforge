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
    batch_size: int = 64,
    trust_remote_code: bool = True,
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
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    prompt_texts: list[str] = []
    input_lengths: list[int] = []
    for prompt in prompts:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
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
    pbar = tqdm(total=total, desc=f"vllm-generate:{condition}", dynamic_ncols=True)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_prompts = prompt_texts[start:end]
        if logger:
            logger.event(
                "vllm_generation_batch_start",
                {
                    "condition": condition,
                    "start": start,
                    "end": end,
                    "total": total,
                    "batch_size": end - start,
                    "max_new_tokens": max_new_tokens,
                    "max_length": max_length,
                },
            )
        outputs = llm.generate(batch_prompts, sampling, use_tqdm=False)
        for offset, out in enumerate(outputs):
            i = start + offset
            response = out.outputs[0].text if out.outputs else ""
            row = {
                "i": i,
                "prompt": prompts[i],
                "response": response,
                "keyword_refusal": is_refusal(response),
                "input_tokens": input_lengths[i],
                "max_new_tokens": max_new_tokens,
                "max_length": max_length,
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
                        "keyword_refusal": row["keyword_refusal"],
                        "backend": "vllm",
                    },
                )
            pbar.update(1)
            pbar.set_postfix(chars=len(response), refusal=row["keyword_refusal"])
        if logger:
            logger.event(
                "vllm_generation_batch_done",
                {"condition": condition, "done": end, "total": total},
            )
    pbar.close()
    return rows, "vllm"
