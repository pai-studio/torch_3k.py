import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
from torch_1k import allclose


def test_global_max_value_and_backward():
    x = torch.tensor([1.0, 3.0, 2.0], requires_grad=True)

    y = torch.max(x)
    y.backward()

    assert y.item() == 3.0
    assert allclose(x.grad, [0.0, 1.0, 0.0])


def test_dim_max_returns_values_indices_and_backprops():
    x = torch.tensor(
        [[1.0, 3.0, 2.0], [4.0, 0.0, 2.0]],
        requires_grad=True,
    )

    result = x.max(dim=1)
    values, indices = result
    values.sum().backward()

    assert np.allclose(result.values.numpy(), [3.0, 4.0])
    assert np.allclose(values.numpy(), [3.0, 4.0])
    assert np.allclose(indices.numpy(), [1, 0])
    assert indices.requires_grad is False
    assert allclose(x.grad, [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])


def test_torch_max_dim_keepdim():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 2.0]])

    result = torch.max(x, dim=1, keepdim=True)

    assert result.values.shape == (2, 1)
    assert result.indices.shape == (2, 1)
    assert np.allclose(result.values.numpy(), [[3.0], [4.0]])
    assert np.allclose(result.indices.numpy(), [[1], [0]])


def test_elementwise_torch_max_and_backward():
    x = torch.tensor([1.0, 5.0, -1.0], requires_grad=True)
    y = torch.tensor([2.0, 3.0, 0.0], requires_grad=True)

    z = torch.max(x, y)
    z.sum().backward()

    assert np.allclose(z.detach().numpy(), [2.0, 5.0, 0.0])
    assert allclose(x.grad, [0.0, 1.0, 0.0])
    assert allclose(y.grad, [1.0, 0.0, 1.0])


def test_max_on_plain_data_does_not_build_graph():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = x.max()

    assert y.requires_grad is False
    assert y.creator is None
