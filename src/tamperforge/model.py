"""Model + SAE loading, device selection, residual-stream capture."""

from __future__ import annotations

import os

import torch

MODEL_ID = "google/gemma-3-1b-it"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
SAE_ID = "layer_13_width_16k_l0_medium"


def qwen_thinking_mode(default: str = "off") -> str:
    """Return the requested Qwen3 thinking mode: off, on, or default."""
    mode = os.environ.get("TF_QWEN_THINKING", default).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "none": "off",
        "1": "on",
        "true": "on",
        "yes": "on",
        "auto": "default",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"off", "on", "default"}:
        raise ValueError(f"invalid TF_QWEN_THINKING={mode!r}; use off, on, or default")
    return mode


def apply_chat_template_no_think(tok, messages, **kwargs):
    """Apply chat template with explicit Qwen3 thinking control when supported.

    Historical TamperForge Qwen runs used no-thinking mode, so the default stays
    off. Set TF_QWEN_THINKING=on for Qwen3 reasoning-mode experiments, or
    TF_QWEN_THINKING=default to let the tokenizer decide.
    """
    mode = qwen_thinking_mode("off")
    if mode == "default":
        return tok.apply_chat_template(messages, **kwargs)
    try:
        return tok.apply_chat_template(messages, enable_thinking=(mode == "on"), **kwargs)
    except TypeError:
        return tok.apply_chat_template(messages, **kwargs)


def pick_device(prefer: str | None = None) -> str:
    """Return the best available device: cuda > mps > cpu (or *prefer* if valid)."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str = MODEL_ID, device: str | None = None):
    """Load tokenizer + causal LM in bf16 on the chosen device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"torch_dtype": torch.bfloat16}
    # gemma-3 NaNs in bf16 training under sdpa/flash (attention soft-capping);
    # eager is the stable path. TF_ATTN_IMPL env overrides for any model.
    attn = os.environ.get("TF_ATTN_IMPL") or ("eager" if "gemma" in model_id.lower() else None)
    if attn:
        kwargs["attn_implementation"] = attn
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model = model.to(device).eval()
    return model, tok, device


def load_sae(release: str = SAE_RELEASE, sae_id: str = SAE_ID, device: str | None = None):
    """Load a Gemma Scope 2 SAE. Returns the SAE object directly (new sae_lens API)."""
    from sae_lens import SAE

    device = pick_device(device)
    return SAE.from_pretrained(release=release, sae_id=sae_id, device=device)


@torch.no_grad()
def capture_residuals(
    model,
    tok,
    prompts: list[str],
    layers: list[int],
    device: str,
    use_chat_template: bool = True,
    adapter=None,
    adapter_layer: int | None = None,
    batch_size: int = 32,
) -> dict[int, torch.Tensor]:
    """Mean last-token residual-stream activation at EACH of *layers* per prompt.

    Returns ``{layer: [n_prompts, d_model]}`` (float32). Hooking every layer costs
    the same forwards as hooking one — a single forward already computes them all —
    so per-layer refusal directions are as cheap as a single-layer estimate.

    Batched with RIGHT padding, which makes this bit-equivalent to the old one-prompt-
    at-a-time loop rather than merely close. Under causal attention a real token at
    position i attends only to 0..i, so trailing pad tokens cannot influence any real
    position, and every sequence keeps the same positions it had unpadded (no RoPE
    shift — the reason LEFT padding would need explicit position_ids). We then gather
    each row at its own true final index instead of a shared -1.

    This is the hot path for direction estimation: a version_A refresh captures 768
    prompts across every layer, and at batch 1 that is 768 kernel-launch-bound forwards
    burning a few percent of the GPU. `batch_size` only trades memory for speed; drop
    it if a long-prompt corpus makes the padded batch too wide.
    """
    acts: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    captured: dict[int, torch.Tensor] = {}
    handles = []
    adapter_handle = None

    if adapter is not None:
        from tamperforge.adapter import make_adapter_hook

        hook_layer = layers[0] if adapter_layer is None else adapter_layer
        adapter_handle = model.model.layers[hook_layer].register_forward_hook(
            make_adapter_hook(adapter)
        )

    def _mk(li):
        def hook(module, inp, out):  # noqa: ARG001
            captured[li] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    for li in layers:
        handles.append(model.model.layers[li].register_forward_hook(_mk(li)))
    try:
        texts = [
            apply_chat_template_no_think(
                tok, [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            ) if use_chat_template else p
            for p in prompts
        ]
        if tok.pad_token_id is None:          # some tokenizers ship without one
            tok.pad_token = tok.eos_token
        old_side, tok.padding_side = tok.padding_side, "right"
        try:
            for i in range(0, len(texts), max(1, batch_size)):
                chunk = texts[i: i + max(1, batch_size)]
                enc = tok(chunk, return_tensors="pt", padding=True).to(device)
                model(**enc)
                # each row's own last real token, not a shared -1 over the padded width
                last = enc["attention_mask"].sum(1) - 1
                rows = torch.arange(len(chunk), device=last.device)
                for li in layers:
                    acts[li].append(captured[li][rows, last].float().cpu())
        finally:
            tok.padding_side = old_side
    finally:
        for h in handles:
            h.remove()
        if adapter_handle is not None:
            adapter_handle.remove()
    return {li: torch.cat(v) for li, v in acts.items()}


def capture_residual(
    model,
    tok,
    prompts: list[str],
    layer: int,
    device: str,
    use_chat_template: bool = True,
    adapter=None,
    adapter_layer: int | None = None,
) -> torch.Tensor:
    """Mean last-token residual-stream activation at *layer* for each prompt.

    Returns ``[n_prompts, d_model]`` (float32). Used to derive empirical
    refusal directions (mean harmful − mean harmless).
    """
    return capture_residuals(model, tok, prompts, [layer], device,
                             use_chat_template, adapter, adapter_layer)[layer]
