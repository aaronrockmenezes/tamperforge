"""Rank-k ablation must reduce to the original rank-1 outer-product form exactly."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from v11_surgical_ablation import READ_P, WRITE_P, _attack_  # noqa: E402


class _Stub:
    """Minimal stand-in exposing named_parameters() with the real weight key format."""

    def __init__(self, d_model=16, n_layers=2):
        self.p = {}
        for li in range(n_layers):
            for name in READ_P + WRITE_P:
                self.p[f"model.layers.{li}.{name}.weight"] = torch.nn.Parameter(
                    torch.randn(d_model, d_model))

    def named_parameters(self):
        return self.p.items()


def test_rank1_matches_outer_product_form():
    torch.manual_seed(0)
    d_model = 16
    d = torch.randn(d_model)
    d = d / d.norm()

    got, want = _Stub(d_model), _Stub(d_model)
    for k, v in want.p.items():          # identical starting weights
        got.p[k].data.copy_(v.data)

    _attack_(got, d.unsqueeze(0), layers=[0, 1])

    # the original implementation, verbatim
    for li in [0, 1]:
        for name in READ_P:
            W = want.p[f"model.layers.{li}.{name}.weight"]
            W.data.copy_(W.float() - torch.outer(W.float() @ d, d))
        for name in WRITE_P:
            W = want.p[f"model.layers.{li}.{name}.weight"]
            W.data.copy_(W.float() - torch.outer(d, d @ W.float()))

    for k in want.p:
        assert torch.allclose(got.p[k], want.p[k], atol=1e-6), k


def test_rank_k_projector_is_idempotent_and_kills_the_span():
    torch.manual_seed(1)
    d_model, k = 16, 4
    R = torch.linalg.qr(torch.randn(d_model, k))[0].T      # [k, d_model], orthonormal

    m = _Stub(d_model)
    _attack_(m, R, layers=[0])
    once = {kk: v.detach().clone() for kk, v in m.p.items()}
    _attack_(m, R, layers=[0])                              # applying twice changes nothing
    for kk in once:
        assert torch.allclose(once[kk], m.p[kk], atol=1e-6), kk

    # read matrices must have no component along the span left
    W = m.p[f"model.layers.0.{READ_P[0]}.weight"].float()
    assert (W @ R.T).abs().max() < 1e-5


if __name__ == "__main__":
    test_rank1_matches_outer_product_form()
    test_rank_k_projector_is_idempotent_and_kills_the_span()
    print("ok")
