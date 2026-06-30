"""Refusal-direction extraction.

Two sources:
  - empirical: mean(h|harmful) − mean(h|harmless)  ← the real attack (Arditi)
  - sae:       a Gemma Scope feature decoder row    ← mechanistic identification

The empirical direction is what an *adaptive* attacker recomputes on the
released model (ROADMAP P2), so it is the honest one to abliterate in P1.
"""

from __future__ import annotations

import torch

from .model import capture_residual


def empirical_refusal_direction(
    model,
    tok,
    harmful: list[str],
    harmless: list[str],
    layer: int,
    device: str,
    adapter=None,
    adapter_layer: int | None = None,
) -> torch.Tensor:
    """mean(h|harmful) − mean(h|harmless) at *layer*, unit-normed ``[d_model]``."""
    h_harm = capture_residual(
        model, tok, harmful, layer, device, adapter=adapter, adapter_layer=adapter_layer
    ).mean(0)
    h_safe = capture_residual(
        model, tok, harmless, layer, device, adapter=adapter, adapter_layer=adapter_layer
    ).mean(0)
    d = (h_harm - h_safe)
    return d / d.norm().clamp(min=1e-8)


def sae_feature_directions(sae_W_dec: torch.Tensor, feat_ids: list[int]) -> torch.Tensor:
    """Stack unit-norm SAE decoder rows for *feat_ids* → ``[n, d_model]``."""
    dirs = sae_W_dec[feat_ids].float()
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
