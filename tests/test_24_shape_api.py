import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch_1k as torch
import torch_1k.nn as nn
from torch_1k import allclose


def test_tensor_view_backward_restores_original_shape():
    x = torch.tensor(np.arange(6).reshape(2, 3).astype("float32"),
                     requires_grad=True)

    y = x.view(3, -1)
    y.sum().backward()

    assert y.shape == (3, 2)
    assert x.grad.shape == (2, 3)
    assert allclose(x.grad, np.ones((2, 3)))


def test_unsqueeze_tensor_method_and_top_level():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    y = x.unsqueeze(1)
    z = torch.unsqueeze(y, -1)
    z.sum().backward()

    assert y.shape == (2, 1, 2)
    assert z.shape == (2, 1, 2, 1)
    assert allclose(x.grad, np.ones((2, 2)))


def test_squeeze_all_dim_and_negative_dim():
    x = torch.tensor(np.arange(6).reshape(1, 2, 1, 3).astype("float32"),
                     requires_grad=True)

    y = x.squeeze()
    z = x.squeeze(dim=-2)
    same = x.squeeze(dim=1)
    (y.sum() + z.sum()).backward()

    assert y.shape == (2, 3)
    assert z.shape == (1, 2, 3)
    assert same.shape == x.shape
    assert allclose(x.grad, np.ones(x.shape) * 2)


def test_top_level_squeeze_axis_tuple():
    x = torch.tensor(np.arange(6).reshape(1, 2, 1, 3).astype("float32"))

    y = torch.squeeze(x, axis=(0, 2))

    assert y.shape == (2, 3)
    assert np.allclose(y.numpy(), np.arange(6).reshape(2, 3))


def test_flatten_tensor_method_top_level_and_nn_module():
    x = torch.tensor(np.arange(24).reshape(2, 3, 4).astype("float32"),
                     requires_grad=True)

    y = x.flatten(start_dim=1)
    z = torch.flatten(x, start_dim=0, end_dim=1)
    module_y = nn.Flatten(start_dim=1)(x)
    (y.sum() + module_y.sum()).backward()

    assert y.shape == (2, 12)
    assert z.shape == (6, 4)
    assert module_y.shape == (2, 12)
    assert np.allclose(y.numpy(), module_y.numpy())
    assert allclose(x.grad, np.ones(x.shape) * 2)


def test_shape_api_rejects_bad_dims():
    x = torch.zeros(2, 3)

    with pytest.raises(IndexError):
        x.unsqueeze(4)
    with pytest.raises(ValueError):
        x.flatten(start_dim=1, end_dim=0)


def test_cuda_shape_api_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor(np.arange(6).reshape(1, 2, 3).astype("float32"),
                     device="cuda", requires_grad=True)
    y = x.squeeze(0).unsqueeze(-1).flatten(start_dim=0)
    y.sum().backward()

    assert y.device == "cuda"
    assert y.shape == (6, 1)
    assert x.grad.device == "cuda"
    assert np.allclose(x.grad.cpu().numpy(), np.ones((1, 2, 3)))
