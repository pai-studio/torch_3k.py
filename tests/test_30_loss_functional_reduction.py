import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch_1k as torch
import torch_1k.nn as nn
import torch_1k.nn.functional as F


def _cross_entropy_expected(logits, target):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    losses = -np.log(probs[np.arange(target.shape[0]), target])
    grad = probs.copy()
    grad[np.arange(target.shape[0]), target] -= 1
    return losses, grad


def test_cross_entropy_functional_reductions_match_manual_values():
    logits_data = np.array([
        [2.0, 0.0, -1.0],
        [0.5, 1.0, 3.0],
        [-1.0, 2.0, 0.0],
    ])
    target_data = np.array([0, 2, 1])
    logits = torch.tensor(logits_data, requires_grad=True)
    target = torch.tensor(target_data)
    expected_losses, _ = _cross_entropy_expected(logits_data, target_data)

    loss_none = F.cross_entropy(logits, target, reduction="none")
    loss_sum = F.cross_entropy(logits, target, reduction="sum")
    loss_mean = F.cross_entropy(logits, target)

    assert np.allclose(loss_none.numpy(), expected_losses)
    assert np.allclose(loss_sum.item(), expected_losses.sum())
    assert np.allclose(loss_mean.item(), expected_losses.mean())


def test_cross_entropy_reduction_controls_gradient_scale():
    logits_data = np.array([[2.0, 0.0], [0.0, 2.0]])
    target_data = np.array([0, 1])
    _, expected_grad = _cross_entropy_expected(logits_data, target_data)

    logits_sum = torch.tensor(logits_data, requires_grad=True)
    F.cross_entropy(logits_sum, torch.tensor(target_data), reduction="sum").backward()
    assert torch.allclose(logits_sum.grad, expected_grad)

    logits_mean = torch.tensor(logits_data, requires_grad=True)
    F.cross_entropy(logits_mean, torch.tensor(target_data), reduction="mean").backward()
    assert torch.allclose(logits_mean.grad, expected_grad / target_data.shape[0])

    logits_none = torch.tensor(logits_data, requires_grad=True)
    F.cross_entropy(logits_none, torch.tensor(target_data),
                    reduction="none").sum().backward()
    assert torch.allclose(logits_none.grad, expected_grad)


def test_cross_entropy_module_accepts_reduction_argument():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    target = torch.tensor([0, 1])

    loss = nn.CrossEntropyLoss(reduction="none")(logits, target)

    assert loss.shape == (2,)
    loss.sum().backward()
    assert logits.grad.shape == logits.shape


def test_mse_loss_functional_reductions_and_target_grad():
    pred_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    target_data = np.array([[0.0, 1.0], [1.0, 1.0]])
    expected_none = (pred_data - target_data) ** 2

    pred = torch.tensor(pred_data, requires_grad=True)
    target = torch.tensor(target_data, requires_grad=True)
    loss_none = F.mse_loss(pred, target, reduction="none")
    loss_sum = F.mse_loss(pred, target, reduction="sum")

    assert np.allclose(loss_none.numpy(), expected_none)
    assert np.allclose(loss_sum.item(), expected_none.sum())

    loss_mean = F.mse_loss(pred, target, reduction="mean")
    loss_mean.backward()

    expected_grad = 2.0 * (pred_data - target_data) / pred_data.size
    assert torch.allclose(pred.grad, expected_grad)
    assert torch.allclose(target.grad, -expected_grad)


def test_invalid_reduction_is_rejected():
    with pytest.raises(ValueError):
        nn.CrossEntropyLoss(reduction="median")
    with pytest.raises(ValueError):
        F.mse_loss(torch.tensor([1.0]), torch.tensor([1.0]),
                   reduction="median")


def test_cross_entropy_cuda_if_available():
    if not torch.cuda.is_available():
        return

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], device="cuda",
                          requires_grad=True)
    target = torch.tensor([0, 1], device="cuda")

    loss = F.cross_entropy(logits, target)
    loss.backward()

    assert loss.device == "cuda"
    assert logits.grad.device == "cuda"
    assert F.cross_entropy(logits, target, reduction="none").device == "cuda"
