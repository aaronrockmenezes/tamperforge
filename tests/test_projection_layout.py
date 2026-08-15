"""Architecture resolver smoke. Run: PYTHONPATH=src python tests/test_projection_layout.py"""

import torch

from tamperforge import (
    abliterate_model_inplace,
    abliteration_parameter_layout,
    decoder_layers,
)


def linear(parent, name, d, out=None):
    setattr(parent, name, torch.nn.Linear(d, out or d, bias=False))


class StandardLayer(torch.nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.self_attn = torch.nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            linear(self.self_attn, name, d)
        self.mlp = torch.nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear(self.mlp, name, d)


class PhiLayer(torch.nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.self_attn = torch.nn.Module()
        linear(self.self_attn, "qkv_proj", d, 3 * d)
        linear(self.self_attn, "o_proj", d)
        self.mlp = torch.nn.Module()
        linear(self.mlp, "gate_up_proj", d, 2 * d)
        linear(self.mlp, "down_proj", 2 * d, d)


class LinearAttentionLayer(torch.nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.linear_attn = torch.nn.Module()
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            linear(self.linear_attn, name, d)
        linear(self.linear_attn, "out_proj", d)
        self.mlp = torch.nn.Module()
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear(self.mlp, name, d)


class Model(torch.nn.Module):
    def __init__(self, layer, nested=False):
        super().__init__()
        self.model = torch.nn.Module()
        root = self.model
        if nested:
            root.language_model = torch.nn.Module()
            root = root.language_model
        root.layers = torch.nn.ModuleList([layer])


def main():
    standard = Model(StandardLayer())
    assert len(decoder_layers(standard)) == 1
    row = abliteration_parameter_layout(standard)[0]
    assert (len(row["read"]), len(row["write"])) == (5, 2)

    phi = Model(PhiLayer())
    row = abliteration_parameter_layout(phi)[0]
    assert (len(row["read"]), len(row["write"])) == (2, 2)
    d = torch.randn(8)
    d /= d.norm()
    abliterate_model_inplace(phi, d, [0])
    for _, name in row["read"]:
        assert torch.allclose(dict(phi.named_parameters())[name].float() @ d, torch.zeros_like(
            dict(phi.named_parameters())[name].float() @ d), atol=2e-5)
    for _, name in row["write"]:
        assert torch.allclose(d @ dict(phi.named_parameters())[name].float(), torch.zeros_like(
            d @ dict(phi.named_parameters())[name].float()), atol=2e-5)

    hybrid = Model(LinearAttentionLayer(), nested=True)
    row = abliteration_parameter_layout(hybrid)[0]
    assert (len(row["read"]), len(row["write"])) == (6, 2)
    print("projection layout OK: standard, fused Phi, nested hybrid")


if __name__ == "__main__":
    main()
