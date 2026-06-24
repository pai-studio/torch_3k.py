import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch_1k as torch
from torch_1k import allclose


def test_cat_dim0_forward_and_backward():
    x1 = torch.tensor([[1.0, 2.0]], requires_grad=True)
    x2 = torch.tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)

    y = torch.cat([x1, x2], dim=0)
    y.sum().backward()

    assert y.shape == (3, 2)
    assert np.allclose(y.detach().numpy(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert allclose(x1.grad, [[1.0, 1.0]])
    assert allclose(x2.grad, [[1.0, 1.0], [1.0, 1.0]])


def test_cat_dim1_and_negative_dim_backward():
    x1 = torch.tensor([[1.0], [2.0]], requires_grad=True)
    x2 = torch.tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)

    y = torch.cat([x1, x2], dim=-1)
    loss = (y * y).sum()
    loss.backward()

    assert y.shape == (2, 3)
    assert np.allclose(y.detach().numpy(), [[1.0, 3.0, 4.0], [2.0, 5.0, 6.0]])
    assert allclose(x1.grad, [[2.0], [4.0]])
    assert allclose(x2.grad, [[6.0, 8.0], [10.0, 12.0]])


def test_concat_and_concatenate_aliases():
    x1 = torch.tensor([[1.0], [2.0]])
    x2 = torch.tensor([[3.0], [4.0]])

    y1 = torch.concat([x1, x2], dim=1)
    y2 = torch.concatenate([x1, x2], axis=1)

    assert np.allclose(y1.numpy(), [[1.0, 3.0], [2.0, 4.0]])
    assert np.allclose(y2.numpy(), [[1.0, 3.0], [2.0, 4.0]])


def test_cat_accepts_plain_inputs_and_tracks_required_grad_only():
    x = torch.tensor([[1.0]], requires_grad=True)

    y = torch.cat([x, [[2.0]]], dim=0)
    y.sum().backward()

    assert y.shape == (2, 1)
    assert allclose(x.grad, [[1.0]])


def test_cat_rejects_empty_and_mismatched_shapes():
    with pytest.raises(ValueError):
        torch.cat([], dim=0)

    x1 = torch.zeros(2, 3)
    x2 = torch.zeros(2, 4)
    with pytest.raises(ValueError):
        torch.cat([x1, x2], dim=0)


def test_cat_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x1 = torch.tensor([[1.0, 2.0]], device="cuda", requires_grad=True)
    x2 = torch.tensor([[3.0, 4.0]], device="cuda", requires_grad=True)
    y = torch.cat([x1, x2], dim=0)
    y.sum().backward()

    assert y.device == "cuda"
    assert x1.grad.device == "cuda"
    assert x2.grad.device == "cuda"
    assert np.allclose(y.cpu().numpy(), [[1.0, 2.0], [3.0, 4.0]])
