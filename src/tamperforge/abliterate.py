"""Weight-space abliteration utilities (the attack we defend against).

Implements the rank-1 projection surgery from Arditi et al. 2024, applied to
either mechanistically-identified SAE feature directions or an empirically
computed refusal direction.

For each direction d (unit-norm, d_model-dimensional):
  - READ matrices  (W maps residual→hidden):  W ← W − (W @ d) ⊗ d
  - WRITE matrices (W maps hidden→residual):  W ← W − d ⊗ (d @ W)
"""

from __future__ import annotations

import torch


def project_out_read(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Project *d* out of a READ matrix ``[hidden, d_model]`` (q/k/v/gate/up)."""
    d = d / d.norm().clamp(min=1e-8)
    return W - torch.outer(W @ d, d)


def project_out_write(W: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Project *d* out of a WRITE matrix ``[d_model, hidden]`` (o_proj/down_proj)."""
    d = d / d.norm().clamp(min=1e-8)
    return W - torch.outer(d, d @ W)


def abliterate_model_inplace(
    model,
    directions: torch.Tensor,
    layers: list[int],
) -> None:
    """Abliterate the given direction(s) from the model in-place.

    Projects each direction out of all attention and MLP weight matrices in the
    specified layers. Modifies the model in-place; nothing saved to disk.

    Args:
        model: ``AutoModelForCausalLM`` instance (already on device).
        directions: Unit-norm directions ``[n_dirs, d_model]`` (or ``[d_model]``).
        layers: Layer indices to abliterate (e.g. ``[13]`` or ``list(range(26))``).
    """
    dev = next(model.parameters()).device
    if directions.dim() == 1:
        directions = directions.unsqueeze(0)
    dirs = [(directions[i].float().to(dev) / directions[i].float().norm().clamp(min=1e-8))
            for i in range(directions.shape[0])]

    print(f"[abliterate] {len(dirs)} direction(s), "
          f"{len(layers)} layer(s): {layers[0]}..{layers[-1]}")

    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        read_mods = [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
        ]
        write_mods = [layer.self_attn.o_proj, layer.mlp.down_proj]

        for mod in read_mods:
            W = mod.weight.data.float().to(dev)
            for d in dirs:
                W = project_out_read(W, d)
            mod.weight.data = W.to(mod.weight.dtype)

        for mod in write_mods:
            W = mod.weight.data.float().to(dev)
            for d in dirs:
                W = project_out_write(W, d)
            mod.weight.data = W.to(mod.weight.dtype)

    print("[abliterate] done")
