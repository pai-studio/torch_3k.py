import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch as pytorch
import torch_1k as torch
import torch_1k.nn as nn
import torch_1k.nn.functional as F


LOGITS_4D = np.array([
    [
        [[2.0, 0.0], [1.0, -1.0]],
        [[0.0, 1.0], [2.0, 0.5]],
        [[-1.0, 2.0], [0.0, 1.5]],
    ],
    [
        [[0.5, 2.0], [-0.5, 1.0]],
        [[1.5, -1.0], [0.5, 0.0]],
        [[-0.5, 0.0], [2.0, -1.0]],
    ],
], dtype=np.float64)
TARGET_3D = np.array([
    [[0, 2], [1, -100]],
    [[1, 0], [2, 1]],
], dtype=np.int64)
CLASS_WEIGHT = np.array([1.0, 2.0, 0.5], dtype=np.float64)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_cross_entropy_advanced_matches_pytorch_high_dim(reduction):
    logits = torch.tensor(LOGITS_4D, requires_grad=True)
    target = torch.tensor(TARGET_3D)
    weight = torch.tensor(CLASS_WEIGHT)

    loss = F.cross_entropy(
        logits,
        target,
        weight=weight,
        ignore_index=-100,
        reduction=reduction,
        label_smoothing=0.2,
    )
    if reduction == "none":
        loss.sum().backward()
    else:
        loss.backward()

    torch_logits = pytorch.tensor(
        LOGITS_4D, dtype=pytorch.float64, requires_grad=True,
    )
    torch_target = pytorch.tensor(TARGET_3D, dtype=pytorch.int64)
    torch_weight = pytorch.tensor(CLASS_WEIGHT, dtype=pytorch.float64)
    torch_loss = pytorch.nn.functional.cross_entropy(
        torch_logits,
        torch_target,
        weight=torch_weight,
        ignore_index=-100,
        reduction=reduction,
        label_smoothing=0.2,
    )
    if reduction == "none":
        torch_loss.sum().backward()
        assert loss.shape == TARGET_3D.shape
    else:
        torch_loss.backward()

    assert np.allclose(loss.numpy(), torch_loss.detach().numpy())
    assert np.allclose(logits.grad.numpy(), torch_logits.grad.detach().numpy())


def test_ignore_index_zeroes_none_loss_and_gradient_row():
    logits = torch.tensor([
        [2.0, 0.0, -1.0],
        [0.5, 1.0, 3.0],
        [-1.0, 2.0, 0.0],
    ], requires_grad=True)
    target = torch.tensor([0, -100, 1])

    loss = F.cross_entropy(
        logits, target, ignore_index=-100, reduction="none",
    )
    loss.sum().backward()

    assert loss.shape == (3,)
    assert np.allclose(loss.numpy()[1], 0.0)
    assert np.allclose(logits.grad.numpy()[1], np.zeros(3))


def test_cross_entropy_module_and_functional_advanced_args_match():
    logits_data = np.array([
        [2.0, 0.0, -1.0],
        [0.5, 1.0, 3.0],
        [-1.0, 2.0, 0.0],
    ], dtype=np.float64)
    target = torch.tensor([0, -100, 1])
    weight = torch.tensor(CLASS_WEIGHT)

    module_logits = torch.tensor(logits_data, requires_grad=True)
    module_loss = nn.CrossEntropyLoss(
        weight=weight,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.1,
    )(module_logits, target)
    module_loss.backward()

    functional_logits = torch.tensor(logits_data, requires_grad=True)
    functional_loss = F.cross_entropy(
        functional_logits,
        target,
        weight=weight,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.1,
    )
    functional_loss.backward()

    assert np.allclose(module_loss.item(), functional_loss.item())
    assert np.allclose(
        module_logits.grad.numpy(), functional_logits.grad.numpy(),
    )


def test_cross_entropy_rejects_invalid_advanced_args():
    logits = torch.tensor(np.ones((2, 3, 2)))

    with pytest.raises(ValueError):
        F.cross_entropy(logits, torch.tensor([0, 1]))

    with pytest.raises(ValueError):
        F.cross_entropy(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([1]),
            weight=torch.tensor([1.0, 2.0]),
        )

    with pytest.raises(ValueError):
        F.cross_entropy(
            torch.tensor([[1.0, 2.0, 3.0]]),
            torch.tensor([3]),
        )

    with pytest.raises(ValueError):
        nn.CrossEntropyLoss(label_smoothing=1.2)


def test_cross_entropy_advanced_cuda_if_available():
    if not torch.cuda.is_available():
        return

    logits = torch.tensor(LOGITS_4D, device="cuda", requires_grad=True)
    target = torch.tensor(TARGET_3D, device="cuda")
    weight = torch.tensor(CLASS_WEIGHT, device="cuda")

    loss = F.cross_entropy(
        logits,
        target,
        weight=weight,
        ignore_index=-100,
        reduction="none",
        label_smoothing=0.1,
    )
    loss.sum().backward()

    assert loss.device == "cuda"
    assert logits.grad.device == "cuda"
    assert loss.shape == TARGET_3D.shape
    assert np.allclose(loss.cpu().numpy()[0, 1, 1], 0.0)
