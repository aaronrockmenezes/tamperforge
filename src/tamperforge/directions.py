"""Refusal-direction extraction.

Two sources:
  - empirical: mean(h|harmful) − mean(h|harmless)  ← the real attack (Arditi)
  - sae:       a Gemma Scope feature decoder row    ← mechanistic identification

The empirical direction is what an *adaptive* attacker recomputes on the
released model (ROADMAP P2), so it is the honest one to abliterate in P1.
"""

from __future__ import annotations

import torch

from .model import capture_residual, capture_residuals

RANK_ESTIMATORS = ("svd", "arditi_residual", "partitioned")


def refusal_subspaces_from_activations(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    ranks: tuple[int, ...] | list[int],
    estimator: str = "arditi_residual",
) -> dict[int, torch.Tensor]:
    """Build nested rank-k refusal bases from already-captured activations."""
    ranks = tuple(sorted(set(ranks)))
    if not ranks:
        return {}
    if estimator not in RANK_ESTIMATORS:
        raise ValueError(f"unknown rank estimator {estimator!r}; choose {RANK_ESTIMATORS}")
    Hh, Hb = harmful.float(), harmless.float()
    X = Hh - Hb.mean(0, keepdim=True)
    max_rank = max(ranks)
    if max_rank > min(X.shape):
        raise ValueError(f"rank {max_rank} exceeds activation shape {tuple(X.shape)}")
    if estimator == "svd":
        basis = torch.linalg.svd(X, full_matrices=False).Vh[:max_rank]
    elif estimator == "arditi_residual":
        d = Hh.mean(0) - Hb.mean(0)
        d = d / d.norm().clamp(min=1e-9)
        residual = X - (X @ d).unsqueeze(1) * d.unsqueeze(0)
        tail = torch.linalg.svd(residual, full_matrices=False).Vh[:max_rank - 1]
        basis = torch.linalg.qr(torch.cat((d.unsqueeze(0), tail), dim=0).T).Q.T[:max_rank]
    else:
        if min(len(Hh), len(Hb)) < max_rank:
            raise ValueError(f"partitioned estimator needs at least {max_rank} prompts per side")
        raw = torch.stack([
            Hh[i::max_rank].mean(0) - Hb[i::max_rank].mean(0)
            for i in range(max_rank)
        ])
        basis = torch.linalg.qr(raw.T).Q.T[:max_rank]
    return {rank: basis[:rank].contiguous() for rank in ranks}


def capability_subspace_from_activations(
    activations: torch.Tensor, rank: int
) -> torch.Tensor:
    """Top capability PCs from centered activations, returned as orthonormal rows."""
    H = activations.float() - activations.float().mean(0, keepdim=True)
    if rank > min(H.shape):
        raise ValueError(f"capability rank {rank} exceeds activation shape {tuple(H.shape)}")
    return torch.linalg.svd(H, full_matrices=False).Vh[:rank].contiguous()


def surgicalize_refusal_subspace(
    refusal: torch.Tensor, capability: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Remove capability-span components from every refusal basis vector, then QR."""
    R = refusal.float().unsqueeze(0) if refusal.ndim == 1 else refusal.float()
    V = capability.float()
    removed = (R @ V.T) @ V
    overlap = float(removed.norm() / R.norm().clamp(min=1e-9))
    residual = R - removed
    q, diag = torch.linalg.qr(residual.T, mode="reduced")
    if q.shape[1] < R.shape[0] or (diag.diagonal().abs() < 1e-6).any():
        raise ValueError("surgical projection collapsed the refusal subspace rank")
    return q.T[: R.shape[0]].contiguous(), overlap


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


def empirical_refusal_directions(
    model,
    tok,
    harmful: list[str],
    harmless: list[str],
    layers: list[int],
    device: str,
    adapter=None,
    adapter_layer: int | None = None,
) -> dict[int, torch.Tensor]:
    """Per-layer refusal directions, unit-normed, in ONE pass over the prompts.

    ``{layer: [d_model]}``. Same forward cost as estimating a single layer (one
    forward already computes every layer's residual), so this is the cheap way to
    get the per-layer adaptive attack basis — do NOT loop
    ``empirical_refusal_direction`` over layers, that re-runs every prompt per layer.
    """
    h_harm = capture_residuals(model, tok, harmful, layers, device,
                               adapter=adapter, adapter_layer=adapter_layer)
    h_safe = capture_residuals(model, tok, harmless, layers, device,
                               adapter=adapter, adapter_layer=adapter_layer)
    out = {}
    for li in layers:
        d = h_harm[li].mean(0) - h_safe[li].mean(0)
        out[li] = d / d.norm().clamp(min=1e-8)
    return out


def sae_feature_directions(sae_W_dec: torch.Tensor, feat_ids: list[int]) -> torch.Tensor:
    """Stack unit-norm SAE decoder rows for *feat_ids* → ``[n, d_model]``."""
    dirs = sae_W_dec[feat_ids].float()
    return dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def svd_refusal_directions(
    model,
    tok,
    harmful: list[str],
    harmless: list[str],
    layer: int,
    device: str,
    k: int = 4,
    whiten: bool = False,
) -> torch.Tensor:
    """Top-*k* refusal directions via SVD of benign-centered harmful activations.

    A stronger/adaptive attacker than diff-in-means (which is rank-1): captures a
    rank-k refusal SUBSPACE. Mirrors OBLITERATUS's SVD / whitened-SVD estimators.

    - Capture harmful H ``[n_h, d]`` and benign B ``[n_b, d]`` residuals at *layer*.
    - Center harmful on the benign mean: ``X = H − mean(B)``.
    - ``whiten=True``: diagonal whitening by benign std before SVD (covariance-
      normalized extraction; separates the refusal signal from natural variance).
      Simple, stable diagonal approximation to full-covariance whitening.
    - Return the top-*k* right singular vectors (unit-norm, orthonormal) ``[k, d]``.
    """
    H = capture_residual(model, tok, harmful, layer, device).float()   # [n_h, d]
    B = capture_residual(model, tok, harmless, layer, device).float()  # [n_b, d]
    mu_b = B.mean(0)
    X = H - mu_b
    if whiten:
        std = B.std(0).clamp(min=1e-6)
        Xw = X / std
        _, _, Vt = torch.linalg.svd(Xw, full_matrices=False)
        dirs = Vt[:k] / std                       # map singular vectors back to model space
    else:
        _, _, Vt = torch.linalg.svd(X, full_matrices=False)
        dirs = Vt[:k]
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # re-orthonormalize (whitening map can break orthogonality)
    q, _ = torch.linalg.qr(dirs.T)
    return q.T[:k].contiguous()
