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
    assert (d @ full).norm() < (d @ half).norm() < (d @ W).norm()


def test_sampler_only_goes_partial_or_perlayer_when_asked():
    rng = random.Random(0)
    base = [_sample_attack(rng, 16) for _ in range(50)]
    assert all(a is None and not pl for *_, a, pl, _t in base), "v8 default must stay full-strength, shared-d"
    got = [_sample_attack(rng, 16, True, True) for _ in range(200)]
    assert any(a is not None for *_, a, _pl, _t in got), "never sampled alphas"
    assert any(pl for *_, _a, pl, _t in got), "never sampled per-layer"
    assert any(not pl for *_, _a, pl, _t in got), "per-layer must not replace the shared-d (Arditi) case"
    assert any("broad" in t for *_, t in got), "never sampled broad coverage"
    for *_, layers, alphas, _pl, _t in got:
        assert alphas is None or (set(alphas) == set(layers) and all(0 < v <= 1 for v in alphas.values()))


def test_per_layer_directions_differ_from_shared():
    m = _Stub()
    p = "mlp.down_proj"
    key = "model.layers.{}." + p + ".weight"
    dirs = {li: torch.nn.functional.normalize(torch.randn(8), dim=0) for li in range(4)}
    per = _ablated_overrides(m, dirs, [0, 1], [], [p])
    shared = _ablated_overrides(m, dirs[0], [0, 1], [], [p])
    # layer 0 uses dirs[0] in both; layer 1 diverges because per-layer uses its own d
    assert torch.allclose(per[key.format(0)], shared[key.format(0)], atol=1e-5)
    assert not torch.allclose(per[key.format(1)], shared[key.format(1)], atol=1e-3)
    # write-proj ablation removes d from the OUTPUT side, so d @ W is what dies
    W1 = dict(m.named_parameters())[key.format(1)]
    assert (dirs[1] @ per[key.format(1)]).norm() < (dirs[1] @ W1).norm() * 0.01
    # and it kills layer 1's own direction, not the shared one it was never given
    assert (dirs[0] @ per[key.format(1)]).norm() > 1e-3
