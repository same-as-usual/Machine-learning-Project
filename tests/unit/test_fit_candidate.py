"""The training loop must survive non-finite batches without poisoning weights.

On a Colab T4 (transformers 5.x + DeBERTa-v3-small) the preflight batch was
finite but a batch at step ~1 produced NaN gradients; clip_grad_norm_ then
scaled every gradient by a NaN norm and one optimizer step poisoned all
weights. The loop now checks the gradient norm BEFORE stepping, skips bad
batches, and gives up on the encoder (-> candidate fallback) only when bad
batches are systemic.
"""

import argparse

import pandas as pd
import torch
from torch import nn

import manipulens.models.transformer as tr

DIM = 6
PARAMS = {"max_length": 12, "batch_size": 4, "epochs": 1, "lr": 1e-3, "warmup_frac": 0.1}
WEIGHTS = {"intensity": 1.0, "taxonomy": 0.5, "binary": 0.5}
DEVICE = torch.device("cpu")


class _FakeTokenizer:
    def __call__(self, text, truncation, max_length, padding, return_tensors):
        return {
            "input_ids": torch.randint(0, 64, (1, max_length)),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }


class _FlakyModel(nn.Module):
    """Same output contract as MultiTaskHeadlineModel; NaNs on chosen calls.

    Call 1 is always the preflight; training steps are calls 2, 3, ...
    """

    def __init__(self, bad_calls=(), bad_from=None):
        super().__init__()
        self.embed = nn.Embedding(64, 8)
        self.intensity_head = nn.Linear(8, 1)
        self.taxonomy_head = nn.Linear(8, DIM)
        self.binary_head = nn.Linear(8, 1)
        self.bad_calls = set(bad_calls)
        self.bad_from = bad_from
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        self.calls += 1
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (self.embed(input_ids) * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        out = {
            "intensity": self.intensity_head(pooled).squeeze(-1),
            "taxonomy": self.taxonomy_head(pooled),
            "binary": self.binary_head(pooled).squeeze(-1),
        }
        bad = self.calls in self.bad_calls or (
            self.bad_from is not None and self.calls >= self.bad_from
        )
        if bad:
            out = {k: v * float("nan") for k, v in out.items()}
        return out


def _df(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "headline": [f"headline number {i} you won't believe" for i in range(n)],
            "intensity": [((i % 10) / 10.0) for i in range(n)],
            "label": [float(i % 2) for i in range(n)],
        }
    )


def _args() -> argparse.Namespace:
    return argparse.Namespace(model_name=None, smoke=True, max_steps=None, batch_size=None)


def _fit(monkeypatch, model: _FlakyModel, n_rows: int = 12):
    monkeypatch.setattr(tr.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer())
    monkeypatch.setattr(tr, "MultiTaskHeadlineModel", lambda name: model)
    return tr._fit_candidate(
        "fake-encoder", _df(n_rows), _df(8), _args(), PARAMS, WEIGHTS, DEVICE
    )


def test_healthy_model_trains_with_no_skips(monkeypatch):
    fitted = _fit(monkeypatch, _FlakyModel())
    assert fitted is not None
    assert fitted[3]["skipped_steps"] == 0
    assert fitted[3]["steps"] == 3  # 12 rows / batch 4


def test_broken_at_preflight_returns_none(monkeypatch):
    assert _fit(monkeypatch, _FlakyModel(bad_from=1)) is None


def test_single_bad_batch_is_skipped_and_training_completes(monkeypatch):
    model = _FlakyModel(bad_calls={3})  # call 1 = preflight; call 3 = 2nd train step
    fitted = _fit(monkeypatch, model)
    assert fitted is not None
    assert fitted[3]["skipped_steps"] == 1
    assert fitted[3]["steps"] == 3


def test_bad_batch_never_reaches_optimizer(monkeypatch):
    model = _FlakyModel(bad_calls={3})
    before = None

    real_forward = model.forward

    def spying_forward(input_ids, attention_mask):
        nonlocal before
        if model.calls + 1 == 3:  # snapshot weights entering the bad step
            before = model.embed.weight.detach().clone()
        return real_forward(input_ids, attention_mask)

    monkeypatch.setattr(model, "forward", spying_forward)
    assert _fit(monkeypatch, model) is not None
    assert torch.isfinite(model.embed.weight).all()


def test_systemically_broken_after_preflight_returns_none(monkeypatch):
    # Preflight (call 1) is finite, every training batch after it is NaN —
    # exactly the observed Colab T4 failure shape.
    assert _fit(monkeypatch, _FlakyModel(bad_from=2)) is None


def test_bails_after_max_consecutive_bad_steps(monkeypatch):
    # Enough rows that the consecutive-bad bail-out (not end of data) triggers.
    n = (tr.MAX_CONSECUTIVE_BAD_STEPS + 5) * PARAMS["batch_size"]
    assert _fit(monkeypatch, _FlakyModel(bad_from=2), n_rows=n) is None
