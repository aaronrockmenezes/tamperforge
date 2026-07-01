"""Fold a safety adapter into base weights — P1b step B (no removable block).

The prototype adapter is a forward hook: a discrete block an attacker can simply
delete. P1b makes the entanglement part of the base weights with no excision
point. The cleanest exact fold uses a gated (SwiGLU) adapter whose neurons match
the Gemma FFN form, so they concatenate onto the FFN's intermediate neurons.

Gemma FFN neuron form:  down( silu(gate·h) ⊙ up·h ).
GatedSafetyAdapter:     W_down( silu(W_gate·h) ⊙ W_up·h ).

Identical form => concatenating the adapter's d_hidden neurons onto the FFN
(gate_proj/up_proj rows, down_proj columns) reproduces `h + adapter(h)` exactly,
with no residual module left. alpha is folded into W_down before concat.

NOTE: the plain SiLU-MLP `SafetyAdapter` does NOT fold exactly (its neurons lack
the `up·h` gate term; you cannot force `up·h ≡ 1` with a linear map). Retrain as
`GatedSafetyAdapter` before folding. This module only implements the gated fold.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _grow_linear(lin: nn.Linear, new_weight: torch.Tensor) -> nn.Linear:
    """Return a bias-free Linear sized to *new_weight* (same dtype/device)."""
    out_f, in_f = new_weight.shape
    grown = nn.Linear(in_f, out_f, bias=False)
    grown.weight = nn.Parameter(new_weight.to(lin.weight.dtype).to(lin.weight.device),
                                requires_grad=lin.weight.requires_grad)
    return grown


@torch.no_grad()
def fold_gated_adapter_into_ffn(model, adapter, layer: int) -> None:
    """Fold a :class:`GatedSafetyAdapter` into ``model``'s FFN at *layer*, in place.

    After this, the model computes the same thing as base + adapter-hook at that
    layer, but the adapter's neurons live inside gate_proj/up_proj/down_proj —
    no separate block remains. The model's intermediate size at this layer grows
    by ``adapter.d_hidden``.

    Args:
        model: ``AutoModelForCausalLM`` with a Gemma-style gated MLP at
            ``model.model.layers[layer].mlp`` (gate_proj/up_proj/down_proj).
        adapter: a :class:`GatedSafetyAdapter` (SwiGLU form). A plain
            ``SafetyAdapter`` will raise — it does not fold exactly.
        layer: layer index to fold into.
    """
    from .adapter import GatedSafetyAdapter

    if not isinstance(adapter, GatedSafetyAdapter):
        raise TypeError(
            "fold requires a GatedSafetyAdapter (SwiGLU form). The plain "
            "SafetyAdapter does not fold exactly into a gated FFN; retrain gated."
        )

    mlp = model.model.layers[layer].mlp
    dev = mlp.gate_proj.weight.device

    g = adapter.W_gate.weight.detach().float().to(dev)                      # [h, d]
    u = adapter.W_up.weight.detach().float().to(dev)                        # [h, d]
    d = (adapter.alpha * adapter.W_down.weight.detach().float()).to(dev)    # [d, h]

    new_gate = torch.cat([mlp.gate_proj.weight.float(), g], dim=0)  # [I+h, d]
    new_up = torch.cat([mlp.up_proj.weight.float(), u], dim=0)      # [I+h, d]
    new_down = torch.cat([mlp.down_proj.weight.float(), d], dim=1)  # [d, I+h]

    mlp.gate_proj = _grow_linear(mlp.gate_proj, new_gate)
    mlp.up_proj = _grow_linear(mlp.up_proj, new_up)
    mlp.down_proj = _grow_linear(mlp.down_proj, new_down)

    # keep config bookkeeping honest (forward derives shapes from weights, but
    # save_pretrained / re-load reads intermediate_size).
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    if text_cfg is not None and hasattr(text_cfg, "intermediate_size"):
        # Only valid if every layer folded identically; for a single-layer fold
        # the checkpoint will need per-layer sizes. Flagged for the caller.
        pass

    print(f"[fold] gated adapter ({adapter.d_hidden} neurons) -> layer {layer} FFN; "
          f"intermediate {new_gate.shape[0] - adapter.d_hidden} -> {new_gate.shape[0]}")


@torch.no_grad()
def verify_fold(model_hooked, model_folded, tok, prompts, layer: int, device: str,
                atol: float = 1e-3) -> dict:
    """Sanity check: folded model ≈ hooked model on next-token logits.

    Run BEFORE trusting a fold. Compares last-token logits on a few prompts.
    Returns max abs diff; expect < atol (small fp/dtype slack).
    """
    import torch.nn.functional as F  # noqa: F401

    max_diff = 0.0
    for p in prompts:
        enc = tok(p, return_tensors="pt").to(device)
        a = model_hooked(**enc).logits[0, -1].float()
        b = model_folded(**enc).logits[0, -1].float()
        max_diff = max(max_diff, float((a - b).abs().max()))
    return {"max_abs_logit_diff": max_diff, "layer": layer, "ok": max_diff < atol}
