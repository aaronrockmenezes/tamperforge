"""Model + SAE loading, device selection, residual-stream capture."""

from __future__ import annotations

import torch

MODEL_ID = "google/gemma-3-1b-it"
SAE_RELEASE = "gemma-scope-2-1b-it-res"
SAE_ID = "layer_13_width_16k_l0_medium"


def apply_chat_template_no_think(tok, messages, **kwargs):
    """Apply chat template with Qwen3 thinking disabled when supported."""
    try:
        return tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
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
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()
    return model, tok, device


def load_sae(release: str = SAE_RELEASE, sae_id: str = SAE_ID, device: str | None = None):
    """Load a Gemma Scope 2 SAE. Returns the SAE object directly (new sae_lens API)."""
    from sae_lens import SAE

    device = pick_device(device)
    return SAE.from_pretrained(release=release, sae_id=sae_id, device=device)


@torch.no_grad()
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
    acts: list[torch.Tensor] = []
    captured = {}
    adapter_handle = None

    if adapter is not None:
        from tamperforge.adapter import make_adapter_hook

        hook_layer = layer if adapter_layer is None else adapter_layer
        adapter_handle = model.model.layers[hook_layer].register_forward_hook(
            make_adapter_hook(adapter)
        )

    def hook(module, inp, out):  # noqa: ARG001
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        for p in prompts:
            if use_chat_template:
                text = apply_chat_template_no_think(
                    tok,
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
            else:
                text = p
            ids = tok(text, return_tensors="pt").to(device)
            model(**ids)
            acts.append(captured["h"][0, -1].float().cpu())
    finally:
        handle.remove()
        if adapter_handle is not None:
            adapter_handle.remove()
    return torch.stack(acts)
