import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch as pytorch
import torch_1k as torch


def test_min_global_splits_gradient_between_ties():
    data = np.array([1.0, -2.0, -2.0, 3.0])
    x = torch.tensor(data, requires_grad=True)
    y = torch.min(x)
    y.backward()

    tx = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    ty = pytorch.min(tx)
    ty.backward()

    assert np.allclose(y.numpy(), ty.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())


def test_min_dim_returns_values_indices_and_argmin_gradient():
    data = np.array([[1.0, -2.0, -2.0], [3.0, 0.0, -1.0]])
    x = torch.tensor(data, requires_grad=True)
    result = torch.min(x, dim=1)
    result.values.sum().backward()

    tx = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    expected = pytorch.min(tx, dim=1)
    expected.values.sum().backward()

    assert np.allclose(result.values.numpy(), expected.values.detach().numpy())
    assert np.allclose(result.indices.numpy(), expected.indices.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())


def test_min_keepdim_and_negative_dim_match_pytorch():
    data = np.array([
        [[1.0, -3.0], [-3.0, 2.0]],
        [[-4.0, -4.0], [1.0, -5.0]],
    ])
    x = torch.tensor(data, requires_grad=True)
    result = torch.min(x, dim=-1, keepdim=True)
    result.values.sum().backward()

    tx = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    expected = pytorch.min(tx, dim=-1, keepdim=True)
    expected.values.sum().backward()

    assert result.values.shape == expected.values.shape
    assert result.indices.shape == expected.indices.shape
    assert np.allclose(result.values.numpy(), expected.values.detach().numpy())
    assert np.allclose(result.indices.numpy(), expected.indices.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())


def test_tensor_min_method_matches_top_level():
    data = np.array([[1.0, -2.0, 3.0], [0.0, -4.0, 5.0]])
    x = torch.tensor(data)

    result = x.min(dim=1)
    expected = torch.min(x, dim=1)

    assert torch.allclose(result.values, expected.values)
    assert torch.allclose(result.indices, expected.indices)
    assert torch.allclose(x.min(axis=1, keepdims=True).values,
                          torch.min(x, axis=1, keepdims=True).values)


def test_elementwise_min_and_minimum_match_pytorch_with_tie_gradients():
    left = np.array([[1.0, 2.0, 2.0], [0.0, -1.0, 3.0]])
    right = np.array([[1.0, 1.0, 3.0], [2.0, -1.0, 1.0]])

    x = torch.tensor(left, requires_grad=True)
    y = torch.tensor(right, requires_grad=True)
    out = torch.min(x, y)
    out.sum().backward()

    tx = pytorch.tensor(left, dtype=pytorch.float64, requires_grad=True)
    ty = pytorch.tensor(right, dtype=pytorch.float64, requires_grad=True)
    expected = pytorch.min(tx, ty)
    expected.sum().backward()

    assert np.allclose(out.numpy(), expected.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())
    assert np.allclose(y.grad.numpy(), ty.grad.detach().numpy())
    assert torch.allclose(torch.minimum(x.detach(), y.detach()),
                          torch.min(x.detach(), y.detach()))


def test_elementwise_min_broadcast_gradient_matches_pytorch():
    left = np.array([[1.0, 2.0, -3.0], [0.0, -1.0, 3.0]])
    right = np.array([1.0, -2.0, -3.0])

    x = torch.tensor(left, requires_grad=True)
    y = torch.tensor(right, requires_grad=True)
    out = torch.minimum(x, y)
    out.sum().backward()

    tx = pytorch.tensor(left, dtype=pytorch.float64, requires_grad=True)
    ty = pytorch.tensor(right, dtype=pytorch.float64, requires_grad=True)
    expected = pytorch.minimum(tx, ty)
    expected.sum().backward()

    assert np.allclose(out.numpy(), expected.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())
    assert np.allclose(y.grad.numpy(), ty.grad.detach().numpy())


def test_min_rejects_tuple_dim_for_reduction():
    x = torch.tensor(np.ones((2, 3, 4)))

    with pytest.raises(TypeError):
        torch.min(x, dim=(1, 2))
    with pytest.raises(TypeError):
        torch.min(x, axis=(1, 2))


def test_min_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor([[1.0, -2.0, -2.0], [3.0, 0.0, -1.0]],
                     device="cuda", requires_grad=True)
    result = torch.min(x, dim=1, keepdim=True)
    result.values.sum().backward()

    assert result.values.device == "cuda"
    assert result.indices.device == "cuda"
    assert x.grad.device == "cuda"
    assert np.allclose(result.values.cpu().numpy(), [[-2.0], [-1.0]])
    assert np.allclose(result.indices.cpu().numpy(), [[1], [2]])
