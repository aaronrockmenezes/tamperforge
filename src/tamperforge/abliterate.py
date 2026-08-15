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

from .model import decoder_layers


READ_PROJECTIONS = {
    "q_proj", "k_proj", "v_proj", "qkv_proj",
    "gate_proj", "up_proj", "gate_up_proj",
    "in_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a",
    "per_layer_input_gate",
}
WRITE_PROJECTIONS = {"o_proj", "out_proj", "down_proj", "per_layer_projection"}


def abliteration_modules(layer, require_both: bool = True):
    """Return named residual-read and residual-write matrices in one decoder layer."""
    read, write = [], []
    for name, module in layer.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if not hasattr(module, "weight") or getattr(module.weight, "ndim", 0) != 2:
            continue
        if leaf in READ_PROJECTIONS:
            read.append((name, module))
        elif leaf in WRITE_PROJECTIONS:
            write.append((name, module))
    if require_both and (not read or not write):
        raise ValueError(
            f"unsupported decoder layer {type(layer).__name__}: "
            f"found {len(read)} read and {len(write)} write projections"
        )
    return read, write


def abliteration_parameter_layout(model):
    """Return architecture-native residual projection parameter names per layer.

    This supports ordinary decoder LMs, nested multimodal text decoders, fused Phi
    projections, and Qwen3.5 linear-attention projections without guessing a fixed
    ``model.layers.*.q_proj`` path.
    """
    cached = getattr(model, "_tamperforge_projection_layout", None)
    if cached is not None:
        return cached
    names = {id(param): name for name, param in model.named_parameters()}
    layout = []
    for index, layer in enumerate(decoder_layers(model)):
        read, write = abliteration_modules(layer, require_both=False)

        def rows(modules):
            out = []
            for local_name, module in modules:
                full_name = names.get(id(module.weight))
                if full_name is None:
                    raise RuntimeError(
                        f"layer {index} projection {local_name} is not a named parameter"
                    )
                out.append((local_name, full_name))
            return tuple(out)

        layout.append({"read": rows(read), "write": rows(write)})
    model._tamperforge_projection_layout = tuple(layout)
    return model._tamperforge_projection_layout


def orthonormalize_directions(directions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return an orthonormal row basis spanning ``directions``.

    ``directions`` may be ``[d_model]`` or ``[n_dirs, d_model]``. Near-zero or
    linearly dependent rows are dropped.
    """
    if directions.dim() == 1:
        directions = directions.unsqueeze(0)
    basis: list[torch.Tensor] = []
    for row in directions.float():
        v = row.clone()
        for b in basis:
            v = v - torch.dot(v, b) * b
        norm = v.norm()
        if norm > eps:
            basis.append(v / norm)
    if not basis:
        raise ValueError("no non-zero directions to orthonormalize")
    return torch.stack(basis, dim=0)


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
        model: text-capable Transformers model (already on device).
        directions: Unit-norm directions ``[n_dirs, d_model]`` (or ``[d_model]``).
        layers: Layer indices to abliterate (e.g. ``[13]`` or ``list(range(26))``).
    """
    dev = next(model.parameters()).device
    dirs = [d.to(dev) for d in orthonormalize_directions(directions)]

    print(f"[abliterate] {len(dirs)} direction(s), "
          f"{len(layers)} layer(s): {layers[0]}..{layers[-1]}")

    model_layers = decoder_layers(model)
    for layer_idx in layers:
        read_named, write_named = abliteration_modules(model_layers[layer_idx])
        read_mods = [module for _, module in read_named]
        write_mods = [module for _, module in write_named]

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


def adapter_wout_directions(adapter, eps: float = 1e-8) -> torch.Tensor:
    """Return orthonormal output directions written by ``adapter.W_out``.

    PyTorch stores linear weights as ``[out_features, in_features]``. Each
    column of ``W_out`` is one residual-stream write direction.
    """
    w = adapter.W_out.weight.detach().float().cpu()
    cols = w.T
    norms = cols.norm(dim=1)
    cols = cols[norms > eps]
    if cols.numel() == 0:
        raise ValueError("adapter W_out has no non-zero output directions")
    return orthonormalize_directions(cols, eps=eps)


def abliterate_adapter_out_inplace(adapter, directions: torch.Tensor) -> None:
    """Project ``directions`` out of the adapter output matrix in-place."""
    dev = adapter.W_out.weight.device
    dirs = [d.to(dev) for d in orthonormalize_directions(directions)]
    W = adapter.W_out.weight.data.float().to(dev)
    for d in dirs:
        W = project_out_write(W, d)
    adapter.W_out.weight.data = W.to(adapter.W_out.weight.dtype)
