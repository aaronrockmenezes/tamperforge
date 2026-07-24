"""v9 idea-2 guard: partial-strength ablation math + sampler contract."""

import random
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from train_tamper_resistant_v8 import _ablated_overrides, _sample_attack  # noqa: E402


class _Stub(nn.Module):
    def __init__(self, n=4, dim=8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            nn.ModuleDict({"mlp": nn.ModuleDict({"down_proj": nn.Linear(dim, dim, bias=False)})})
            for _ in range(n)
        )


def test_alpha_interpolates_between_identity_and_full():
    m, d = _Stub(), torch.nn.functional.normalize(torch.randn(8), dim=0)
    p = "mlp.down_proj"
    W = dict(m.named_parameters())["model.layers.0." + p + ".weight"]
    full = _ablated_overrides(m, d, [0], [], [p])["model.layers.0." + p + ".weight"]
    for a, want in [(0.0, W), (1.0, full)]:
        got = _ablated_overrides(m, d, [0], [], [p], {0: a})["model.layers.0." + p + ".weight"]
        assert torch.allclose(got, want, atol=1e-5), f"alpha={a}"
    half = _ablated_overrides(m, d, [0], [], [p], {0: 0.5})["model.layers.0." + p + ".weight"]
    assert torch.allclose(half, (W + full) / 2, atol=1e-5)
    # the point of alpha: a partial edit leaves refusal-direction energy behind
    assert (full @ d).norm() < (half @ d).norm() < (W @ d).norm()


def test_sampler_only_goes_partial_when_asked():
    rng = random.Random(0)
    assert all(_sample_attack(rng, 16)[3] is None for _ in range(50)), "v8 default must stay full-strength"
    got = [_sample_attack(rng, 16, True) for _ in range(200)]
    assert any(a is not None for *_, a, _tag in got), "partial=True never sampled alphas"
    assert any("broad" in tag for *_, tag in got), "partial=True never sampled broad coverage"
    for *_, layers, alphas, _tag in got:
        assert alphas is None or (set(alphas) == set(layers) and all(0 < v <= 1 for v in alphas.values()))
