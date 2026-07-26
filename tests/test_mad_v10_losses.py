import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from train_mad_v10 import _capability_margin_loss, _target_ce, _target_unlikelihood  # noqa: E402


class _Tokenizer:
    eos_token_id = 0
    vocab_size = 16

    def apply_chat_template(self, messages, **kwargs):
        return "P:" + messages[0]["content"] + "\nA:"

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None,
                 add_special_tokens=False):
        ids = [(ord(c) % (self.vocab_size - 1)) + 1 for c in text]
        if max_length:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _ConstantLM(nn.Module):
    def __init__(self, preferred_token: int, vocab_size: int = 16):
        super().__init__()
        logits = torch.zeros(vocab_size)
        logits[preferred_token] = 8.0
        self.token_logits = nn.Parameter(logits)

    def forward(self, input_ids, labels=None, attention_mask=None):
        logits = self.token_logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
        if labels is None:
            return SimpleNamespace(logits=logits)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return SimpleNamespace(loss=loss, logits=logits)


def test_target_unlikelihood_penalizes_likely_rejected_tokens():
    tok = _Tokenizer()
    rejected = "x"
    preferred = (ord(rejected) % (tok.vocab_size - 1)) + 1
    model = _ConstantLM(preferred)
    loss = _target_unlikelihood(model, tok, [("prompt", rejected)], "cpu")
    assert loss > 5.0


def test_target_ce_rewards_correct_target_probability():
    tok = _Tokenizer()
    target = "x"
    preferred = (ord(target) % (tok.vocab_size - 1)) + 1
    good = _ConstantLM(preferred)
    bad = _ConstantLM((preferred % (tok.vocab_size - 1)) + 1)
    assert _target_ce(good, tok, [("prompt", target)], "cpu") < 0.01
    assert _target_ce(bad, tok, [("prompt", target)], "cpu") > 6.0


def test_capability_margin_fires_when_attack_does_not_hurt_correct_answer():
    tok = _Tokenizer()
    target = "x"
    preferred = (ord(target) % (tok.vocab_size - 1)) + 1
    model = _ConstantLM(preferred)
    loss, clean_ce, attacked_ce, gap = _capability_margin_loss(
        model,
        tok,
        [("prompt", target)],
        "cpu",
        {"token_logits": model.token_logits},
        margin=2.0,
        max_len=64,
    )
    assert clean_ce < 0.01
    assert attacked_ce < 0.01
    assert gap < 0.01
    assert loss > 1.9
