"""The task-degradation loss must measure degradation, not divergence.

The signal it replaces (`--gib-mode argmax`) asks the attacked model to pick DIFFERENT tokens
from the clean model. version_C satisfied that objective and still held GSM8K at 0.3715, and
`gib_ce` scored AUC 0.460 -- chance -- at separating attacked from clean. These tests pin the
properties that failure violated.

Run: pytest tests/test_task_gib_loss.py -q     (or: python tests/test_task_gib_loss.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from train_tamper_resistant_v8 import _task_degradation_loss  # noqa: E402


class _Enc(dict):
    """Stands in for HF BatchEncoding, which supports .to(device)."""
    def to(self, device):  # noqa: ARG002
        return self


class _Tok:
    """Minimal tokeniser: 1 token per word, ids are stable hashes."""
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        ids = [1 + (abs(hash(w)) % 97) for w in text.split()][: (max_length or 10_000)]
        t = torch.tensor([ids]) if ids else torch.zeros((1, 0), dtype=torch.long)
        return _Enc(input_ids=t, attention_mask=torch.ones_like(t))

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "Q : " + msgs[-1]["content"] + " A :"


class _Model(torch.nn.Module):
    """Puts mass on the TRUE next token, scaled by a learnable weight.

    `wrong=True` flips the sign, i.e. a model that actively avoids the right answer. This is
    the property under test: the loss must rise when the model stops getting the task right.
    """

    VOCAB = 128

    def __init__(self, wrong: bool = False) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(-8.0 if wrong else 8.0))

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        oh = torch.nn.functional.one_hot(input_ids, self.VOCAB).float()
        nxt = torch.zeros_like(oh)
        nxt[:, :-1] = oh[:, 1:]          # position t now carries the identity of token t+1
        logits = nxt * self.scale
        out = type("O", (), {})()
        if labels is not None and input_ids.shape[1] > 1:
            out.loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, self.VOCAB),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        else:
            out.loss = logits.sum() * 0.0
        out.logits = logits
        return out


QA = [("what is two plus two", "the answer is four"),
      ("what is three plus one", "the answer is four")]


def test_higher_when_model_gets_answers_wrong():
    """The whole point: a model that fails the task must score HIGHER than one that passes."""
    good = _task_degradation_loss(_Model(wrong=False), _Tok(), QA, "cpu", None)
    bad = _task_degradation_loss(_Model(wrong=True), _Tok(), QA, "cpu", None)
    assert bad > good, f"does not separate: correct={good:.4f} wrong={bad:.4f}"
    assert (bad - good) > 1.0, f"separation too weak to train on: {(bad - good):.4f}"


def test_scores_answer_tokens_only():
    """Prompt tokens are context. If they were scored, prompt length would move the loss."""
    short = [("q", "the answer is four")]
    long = [("q q q q q q q q", "the answer is four")]
    m, tk = _Model(), _Tok()
    a = _task_degradation_loss(m, tk, short, "cpu", None)
    b = _task_degradation_loss(m, tk, long, "cpu", None)
    assert abs(a - b) < 0.35, f"prompt length leaks into the loss: {a:.4f} vs {b:.4f}"


def test_gradients_flow():
    m = _Model()
    _task_degradation_loss(m, _Tok(), QA, "cpu", None).backward()
    g = m.scale.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_empty_answer_does_not_crash_or_nan():
    out = _task_degradation_loss(_Model(), _Tok(), [("q", "")], "cpu", None)
    assert torch.isfinite(out), "empty answer produced a non-finite loss"


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
        print("ok", fn.__name__)
