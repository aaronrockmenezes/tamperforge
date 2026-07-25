"""ABL-v10 guards: explicit shutdown target and isolatable attack profiles."""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from train_tamper_resistant_v8 import (  # noqa: E402
    _attack_metadata,
    _sample_attack_v10,
    _shutdown_loss,
)


class _Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        return "P" + messages[0]["content"]

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [3 + (ord(c) % 5) for c in text]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _ConstantLM(nn.Module):
    def __init__(self, eos_logit: float):
        super().__init__()
        logits = torch.zeros(8)
        logits[2] = eos_logit
        self.token_logits = nn.Parameter(logits)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.token_logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


def test_shutdown_loss_directly_rewards_immediate_eos():
    tok = _Tokenizer()
    good = _ConstantLM(eos_logit=8.0)
    bad = _ConstantLM(eos_logit=-8.0)
    good_loss = _shutdown_loss(
        good, tok, ["benign", "harmful"], "cpu", {"token_logits": good.token_logits}
    )
    bad_loss = _shutdown_loss(
        bad, tok, ["benign", "harmful"], "cpu", {"token_logits": bad.token_logits}
    )
    assert good_loss < 0.01
    assert bad_loss > 8.0


def test_partial_shared_profile_is_banded_and_write_only():
    got = _sample_attack_v10(random.Random(1), 28, "partial_shared", [10, 11, 12], 0.2, 0.6)
    read_p, write_p, layers, alphas, per_layer, tag = got
    assert not read_p
    assert write_p
    assert layers == [10, 11, 12]
    assert set(alphas) == set(layers)
    assert all(0.2 <= value <= 0.6 for value in alphas.values())
    assert not per_layer
    assert "partial_shared" in tag


def test_perlayer_profile_and_metadata_are_explicit():
    got = _sample_attack_v10(random.Random(2), 28, "perlayer_full", [10, 11], 0.2, 0.6)
    _read_p, _write_p, layers, alphas, per_layer, tag = got
    meta = _attack_metadata(tag, layers, alphas, per_layer, "perlayer_full")
    assert alphas is None
    assert per_layer
    assert meta["attack_profile"] == "perlayer_full"
    assert meta["attack_n_layers"] == 2
    assert meta["attack_layer_min"] == 10
    assert meta["attack_layer_max"] == 11
    assert meta["attack_alpha_mean"] == 1.0


def test_mixed_attack_schedule_is_reproducible():
    def schedule():
        rng = random.Random(42)
        return [
            _sample_attack_v10(rng, 28, "mixed", list(range(10, 28)), 0.2, 0.6)[-1]
            for _ in range(20)
        ]

    assert schedule() == schedule()
    assert len(set(schedule())) > 1
