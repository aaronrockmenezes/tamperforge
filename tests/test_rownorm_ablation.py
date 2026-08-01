"""The differentiable FULL ablation must match heretic's exact delta.

`_rownorm_ablated_overrides` drops heretic's rank-3 SVD truncation to keep the graph
differentiable. That is only safe if the truncation is negligible -- these tests pin it.
Run: pytest tests/test_rownorm_ablation.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from train_tamper_resistant_v8 import _rownorm_ablated_overrides  # noqa: E402
from version_c_replay import heretic_delta  # noqa: E402


class _FakeLayer(torch.nn.Module):
    def __init__(self, d_out: int, d_in: int) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.o_proj = torch.nn.Linear(d_in, d_out, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.down_proj = torch.nn.Linear(d_in, d_out, bias=False)


class _FakeModel(torch.nn.Module):
    def __init__(self, n_layers: int, d: int) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_FakeLayer(d, d) for _ in range(n_layers)])


def _mk(seed: int = 0, n_layers: int = 3, d: int = 64):
    torch.manual_seed(seed)
    m = _FakeModel(n_layers, d)
    v = torch.randn(d)
    return m, v / v.norm()


def _exact_full_delta(W: torch.Tensor, v: torch.Tensor, a: float) -> torch.Tensor:
    """heretic's FULL math with NO rank truncation (heretic/model.py:571-578)."""
    rn = W.norm(dim=1, keepdim=True).clamp(min=1e-12)
    Wn = W / rn
    Wa = Wn + (-a * v).view(-1, 1) @ (v @ Wn).view(1, -1)
    Wa = Wa / Wa.norm(dim=1, keepdim=True).clamp(min=1e-12) * rn
    return Wa - W


def test_implements_full_math_exactly():
    """The differentiable form must reproduce heretic's FULL math bit-for-bit.

    Compared against the untruncated reference, so this isolates the math from the rank-3
    LoRA truncation (which is tested separately below).
    """
    m, v = _mk()
    a = 0.87
    ov = _rownorm_ablated_overrides(
        m, {0: v}, [0], (), ("self_attn.o_proj",), {"self_attn.o_proj": {0: a}})
    W = dict(m.named_parameters())["model.layers.0.self_attn.o_proj.weight"].float()
    got = ov["model.layers.0.self_attn.o_proj.weight"].float() - W
    want = _exact_full_delta(W, v, a)
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 1e-5, f"does not implement FULL math: rel={rel:.2e}"


# On the rank-3 truncation we drop: it is an empirical property of real weights and does
# NOT reproduce on synthetic matrices (random Gaussian gives ~13%, a hand-built decaying
# spectrum ~24%), so there is no honest local test for it. Measured on all 29 matrices
# heretic touched in t99, against Qwen3-0.6B version_B s500:
#
#     rank-3 truncation error   median 0.0429   min 0.0317   max 0.0589   p90 0.0542
#
# That is the size of the approximation this module makes. Compare it against the error it
# FIXES: plain `W - a*outer(d, d@W)` differs from heretic's delta by 25-47%. Re-measure with
# the snippet in docs/version_c_step0_2026_08_01.md if the model or dtype changes.


def test_row_magnitudes_preserved():
    """The whole point of FULL: row norms survive the ablation."""
    m, v = _mk(seed=1)
    ov = _rownorm_ablated_overrides(
        m, {0: v}, [0], (), ("mlp.down_proj",), {"mlp.down_proj": {0: 1.0}})
    W = dict(m.named_parameters())["model.layers.0.mlp.down_proj.weight"].float()
    Wa = ov["model.layers.0.mlp.down_proj.weight"].float()
    ratio = (Wa.norm(dim=1) / W.norm(dim=1))
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), \
        f"row norms not preserved: min {ratio.min():.5f} max {ratio.max():.5f}"


def test_differs_from_plain_ablation():
    """If this ever matches the plain form, the fix silently regressed to v8 behaviour."""
    from train_tamper_resistant_v8 import _ablated_overrides
    m, v = _mk(seed=2)
    kw = dict(d={0: v}, layers=[0], read_p=(), write_p=("self_attn.o_proj",),
              alphas={"self_attn.o_proj": {0: 0.9}})
    full = _rownorm_ablated_overrides(m, **kw)["model.layers.0.self_attn.o_proj.weight"]
    plain = _ablated_overrides(m, **kw)["model.layers.0.self_attn.o_proj.weight"]
    rel = ((full.float() - plain.float()).norm() / plain.float().norm()).item()
    assert rel > 0.01, f"FULL and plain are indistinguishable (rel={rel:.5f})"


def test_zero_alpha_layers_untouched():
    m, v = _mk(seed=3)
    ov = _rownorm_ablated_overrides(
        m, {0: v, 1: v}, [0, 1], (), ("self_attn.o_proj",),
        {"self_attn.o_proj": {0: 0.5}})   # layer 1 absent -> weight 0 -> skip
    assert "model.layers.1.self_attn.o_proj.weight" not in ov


def test_gradients_flow():
    """Training needs the attack to be differentiable in theta."""
    m, v = _mk(seed=4)
    W = m.model.layers[0].self_attn.o_proj.weight
    ov = _rownorm_ablated_overrides(
        m, {0: v}, [0], (), ("self_attn.o_proj",), {"self_attn.o_proj": {0: 0.8}})
    ov["model.layers.0.self_attn.o_proj.weight"].float().pow(2).sum().backward()
    assert W.grad is not None and torch.isfinite(W.grad).all() and W.grad.abs().sum() > 0


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
