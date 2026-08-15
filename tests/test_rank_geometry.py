"""Rank/surgical geometry smoke. Run: PYTHONPATH=src python tests/test_rank_geometry.py"""

import torch

from tamperforge import (
    abliterate_model_inplace,
    abliteration_parameter_layout,
    capability_subspace_from_activations,
    refusal_subspaces_from_activations,
    surgicalize_refusal_subspace,
)
from test_projection_layout import Model, StandardLayer


def main():
    torch.manual_seed(7)
    harmful, harmless, capability = (torch.randn(40, 8) for _ in range(3))
    bases = refusal_subspaces_from_activations(
        harmful, harmless, (1, 2, 4), "arditi_residual"
    )
    assert all(torch.allclose(R @ R.T, torch.eye(rank), atol=1e-5)
               for rank, R in bases.items())
    V = capability_subspace_from_activations(capability, 2)
    surgical, _ = surgicalize_refusal_subspace(bases[4], V)
    assert torch.allclose(surgical @ surgical.T, torch.eye(4), atol=1e-5)
    assert torch.allclose(surgical @ V.T, torch.zeros(4, 2), atol=1e-5)

    model = Model(StandardLayer())
    layout = abliteration_parameter_layout(model)[0]
    abliterate_model_inplace(model, bases[2], [0])
    named = dict(model.named_parameters())
    for _, name in layout["read"]:
        assert torch.allclose(named[name].float() @ bases[2].T,
                              torch.zeros(named[name].shape[0], 2), atol=2e-5)
    for _, name in layout["write"]:
        assert torch.allclose(bases[2] @ named[name].float(),
                              torch.zeros(2, named[name].shape[1]), atol=2e-5)
    print("rank geometry OK: shared rank-k basis, all-layer projection, surgical cap removal")


if __name__ == "__main__":
    main()
