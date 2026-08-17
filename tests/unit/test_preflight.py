"""The numerics preflight must reject a broken encoder before training starts.

A DeBERTa-v3 fine-tune on a Colab T4 (transformers 5.x) produced NaN loss from
the very first step with verified-clean data — an environment-specific encoder
bug. The preflight runs one forward+backward on the real device and refuses to
start (or falls back to a known-stable encoder) instead of wasting a full GPU
run training to NaN.
"""

import torch
from torch import nn

from manipulens.models.transformer import preflight_finite

DIM = 6
WEIGHTS = {"intensity": 1.0, "taxonomy": 0.5, "binary": 0.5}
DEVICE = torch.device("cpu")


class _TinyModel(nn.Module):
    """Stand-in for MultiTaskHeadlineModel: same output contract."""

    def __init__(self, poison: str | None = None):
        super().__init__()
        self.embed = nn.Embedding(64, 8)
        self.intensity_head = nn.Linear(8, 1)
        self.taxonomy_head = nn.Linear(8, DIM)
        self.binary_head = nn.Linear(8, 1)
        self.poison = poison
        if poison == "weights":
            with torch.no_grad():
                self.embed.weight.fill_(float("nan"))

    def forward(self, input_ids, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (self.embed(input_ids) * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        out = {
            "intensity": self.intensity_head(pooled).squeeze(-1),
            "taxonomy": self.taxonomy_head(pooled),
            "binary": self.binary_head(pooled).squeeze(-1),
        }
        if self.poison == "outputs":
            out = {k: v * float("nan") for k, v in out.items()}
        return out


def _batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.randint(0, 64, (batch_size, 12)),
        "attention_mask": torch.ones(batch_size, 12, dtype=torch.long),
        "intensity": torch.rand(batch_size),
        "taxonomy": torch.rand(batch_size, DIM),
        "label": torch.randint(0, 2, (batch_size,), dtype=torch.float32),
    }


def test_healthy_model_passes_preflight():
    assert preflight_finite(_TinyModel(), _batch(), WEIGHTS, DEVICE) is True


def test_nan_outputs_fail_preflight():
    assert preflight_finite(_TinyModel(poison="outputs"), _batch(), WEIGHTS, DEVICE) is False


def test_nan_weights_fail_preflight():
    assert preflight_finite(_TinyModel(poison="weights"), _batch(), WEIGHTS, DEVICE) is False


def test_preflight_leaves_no_gradients_behind():
    model = _TinyModel()
    assert preflight_finite(model, _batch(), WEIGHTS, DEVICE) is True
    assert all(p.grad is None for p in model.parameters())


def test_preflight_does_not_change_weights():
    model = _TinyModel()
    before = [p.detach().clone() for p in model.parameters()]
    preflight_finite(model, _batch(), WEIGHTS, DEVICE)
    after = list(model.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))
