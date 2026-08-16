"""NaN/Inf corruption in any target must not poison the whole multitask loss.

A non-finite target (e.g. a single corrupted taxonomy label) used to NaN *every*
task, toast the shared encoder weights, and crash evaluation downstream with
sklearn's "Input contains NaN". Each task must independently contribute only its
finite targets.
"""

import torch

from manipulens.models.transformer import multitask_loss

DIM = 6
WEIGHTS = {"intensity": 1.0, "taxonomy": 0.5, "binary": 0.5}


def _finite_outputs(batch_size: int = 8) -> dict[str, torch.Tensor]:
    return {
        "intensity": torch.randn(batch_size),
        "taxonomy": torch.randn(batch_size, DIM),
        "binary": torch.randn(batch_size),
    }


def _finite_batch(batch_size: int = 8) -> dict[str, torch.Tensor]:
    return {
        "intensity": torch.rand(batch_size),
        "taxonomy": torch.rand(batch_size, DIM),
        "label": torch.randint(0, 2, (batch_size,), dtype=torch.float32),
    }


def test_clean_loss_is_finite():
    loss = multitask_loss(_finite_outputs(), _finite_batch(), WEIGHTS)
    assert torch.isfinite(loss)


def test_nan_taxonomy_target_does_not_poison_loss():
    batch = _finite_batch()
    batch["taxonomy"][3, 2] = float("nan")
    loss = multitask_loss(_finite_outputs(), batch, WEIGHTS)
    assert torch.isfinite(loss)


def test_inf_taxonomy_target_does_not_poison_loss():
    batch = _finite_batch()
    batch["taxonomy"][3, 2] = float("inf")
    loss = multitask_loss(_finite_outputs(), batch, WEIGHTS)
    assert torch.isfinite(loss)


def test_nan_intensity_target_is_masked():
    batch = _finite_batch()
    batch["intensity"][1] = float("nan")
    loss = multitask_loss(_finite_outputs(), batch, WEIGHTS)
    assert torch.isfinite(loss)


def test_nan_binary_label_is_masked():
    batch = _finite_batch()
    batch["label"][5] = float("nan")
    loss = multitask_loss(_finite_outputs(), batch, WEIGHTS)
    assert torch.isfinite(loss)


def test_out_of_range_intensity_target_is_clipped_not_nan():
    batch = _finite_batch()
    batch["intensity"][0] = 7.5  # beyond the sigmoid output range [0, 1]
    loss = multitask_loss(_finite_outputs(), batch, WEIGHTS)
    assert torch.isfinite(loss)


def test_backward_is_finite_and_nullifies_gradients():
    model = torch.nn.Linear(4, 1)
    out = {"intensity": model(torch.randn(8, 4)).squeeze(-1).requires_grad_()}
    out["taxonomy"] = torch.randn(8, DIM, requires_grad=True)
    out["binary"] = torch.randn(8, requires_grad=True)
    batch = _finite_batch()
    batch["taxonomy"][2, 0] = float("nan")
    loss = multitask_loss(out, batch, WEIGHTS)
    loss.backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
