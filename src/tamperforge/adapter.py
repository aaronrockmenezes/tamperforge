"""SafetyAdapter — the prototype entanglement block.

NOTE: this is the *prototype* vehicle (a removable block at one layer). The
*product* (see ROADMAP P1b) is a distributed entanglement pass with no discrete
excision point. The adapter is kept for fast iteration on the MAD property.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class SafetyAdapter(nn.Module):
    """Small MLP inserted into the residual stream::

        h_new = h + alpha * W_out(silu(W_in(h)))

    ``W_in``  [d_hidden, d_model] — detects harmful patterns.
    ``W_out`` [d_model, d_hidden] — writes in language-critical directions.

    Defense property under test (P1): abliterating ``W_out``'s directions from
    base+adapter should destroy language capability as well as safety.
    """

    def __init__(self, d_model: int, d_hidden: int = 256, alpha: float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.alpha = alpha
        self.W_in = nn.Linear(d_model, d_hidden, bias=False)
        self.W_out = nn.Linear(d_hidden, d_model, bias=False)
        self.gate = nn.SiLU()
        nn.init.normal_(self.W_in.weight, std=0.01)
        nn.init.zeros_(self.W_out.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.W_out(self.gate(self.W_in(h)))


class GatedSafetyAdapter(nn.Module):
    """SwiGLU-shaped safety adapter — foldable into a gated MLP (P1b step B)::

        adapter(h) = alpha * W_down( silu(W_gate(h)) * W_up(h) )

    Same neuron form as the Gemma FFN (``down(silu(gate·h) ⊙ up·h)``), so its
    ``d_hidden`` neurons concatenate exactly onto the FFN's intermediate neurons
    via :func:`tamperforge.fold.fold_gated_adapter_into_ffn` — leaving no
    discrete residual block for an attacker to excise. The plain
    :class:`SafetyAdapter` (non-gated SiLU MLP) does NOT fold exactly, which is
    why this variant exists.
    """

    def __init__(self, d_model: int, d_hidden: int = 256, alpha: float = 1.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.alpha = alpha
        self.W_gate = nn.Linear(d_model, d_hidden, bias=False)
        self.W_up = nn.Linear(d_model, d_hidden, bias=False)
        self.W_down = nn.Linear(d_hidden, d_model, bias=False)
        self.act = nn.SiLU()
        nn.init.normal_(self.W_gate.weight, std=0.01)
        nn.init.normal_(self.W_up.weight, std=0.01)
        nn.init.zeros_(self.W_down.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.W_down(self.act(self.W_gate(h)) * self.W_up(h))


def load_adapter(path: str | Path, device: Optional[str] = None) -> tuple[SafetyAdapter, dict]:
    """Load a ``SafetyAdapter`` checkpoint saved by the training script."""
    if device is None:
        device = "cpu"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    adapter = SafetyAdapter(
        d_model=ckpt["d_model"],
        d_hidden=ckpt.get("d_hidden", 256),
        alpha=ckpt.get("alpha", 1.0),
    )
    adapter.load_state_dict(ckpt["state_dict"])
    return adapter.to(device).float().eval(), ckpt


def make_adapter_hook(adapter: SafetyAdapter):
    """Return a ``register_forward_hook`` fn that adds the adapter correction.

    Handles both tuple outputs (Gemma attention layers) and plain tensors.
    """

    def hook(module, inp, out):  # noqa: ARG001
        h = out[0] if isinstance(out, tuple) else out
        correction = adapter(h.float()).to(h.dtype)
        if isinstance(out, tuple):
            return (h + correction,) + out[1:]
        return h + correction

    return hook
