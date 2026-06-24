import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch as pytorch
import torch_1k as torch


def _assert_sort_matches_pytorch(data, dim=-1, descending=False, stable=False):
    weights = np.arange(1, np.size(data) + 1, dtype=np.float64).reshape(
        np.shape(data)
    )

    x = torch.tensor(data, requires_grad=True)
    result = torch.sort(x, dim=dim, descending=descending, stable=stable)
    (result.values * torch.tensor(weights)).sum().backward()

    tx = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    tresult = pytorch.sort(tx, dim=dim, descending=descending, stable=stable)
    (tresult.values * pytorch.tensor(weights, dtype=pytorch.float64)).sum().backward()

    assert np.allclose(result.values.numpy(), tresult.values.detach().numpy())
    assert np.allclose(result.indices.numpy(), tresult.indices.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())
    assert result.values.requires_grad is True
    assert result.indices.requires_grad is False


def test_sort_values_and_indices_match_pytorch():
    data = np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]])

    _assert_sort_matches_pytorch(data, dim=1)


def test_sort_descending_and_negative_dim_match_pytorch():
    data = np.array([
        [[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]],
        [[9.0, 7.0, 8.0], [6.0, 10.0, 11.0]],
    ])

    _assert_sort_matches_pytorch(data, dim=-1, descending=True)


def test_tensor_sort_method_and_stable_signature():
    data = np.array([[2.0, 1.0, 3.0], [6.0, 5.0, 4.0]])
    x = torch.tensor(data, requires_grad=True)

    values, indices = x.sort(dim=1, stable=True)
    (values * torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).sum().backward()

    assert np.allclose(values.numpy(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert np.allclose(indices.numpy(), [[1, 0, 2], [2, 1, 0]])
    assert torch.allclose(x.grad, [[2.0, 1.0, 3.0], [6.0, 5.0, 4.0]])


def test_argsort_top_level_and_tensor_method_match_pytorch():
    data = np.array([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]])
    x = torch.tensor(data, requires_grad=True)
    tx = pytorch.tensor(data, dtype=pytorch.float64)

    indices = torch.argsort(x, dim=1)
    descending = x.argsort(dim=-1, descending=True)

    assert np.allclose(indices.numpy(), pytorch.argsort(tx, dim=1).numpy())
    assert np.allclose(
        descending.numpy(),
        pytorch.argsort(tx, dim=-1, descending=True).numpy(),
    )
    assert indices.requires_grad is False
    assert descending.requires_grad is False


def test_sort_scalar_matches_pytorch():
    x = torch.tensor(3.0, requires_grad=True)
    values, indices = torch.sort(x)
    (values * torch.tensor(2.0)).backward()

    tx = pytorch.tensor(3.0, dtype=pytorch.float64, requires_grad=True)
    tvalues, tindices = pytorch.sort(tx)
    (tvalues * 2.0).backward()

    assert np.allclose(values.numpy(), tvalues.detach().numpy())
    assert np.allclose(indices.numpy(), tindices.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())
    assert np.allclose(torch.argsort(x).numpy(), pytorch.argsort(tx).numpy())


def test_sort_rejects_invalid_dim():
    x = torch.tensor([[1.0, 2.0]])

    with pytest.raises(IndexError):
        torch.sort(x, dim=2)
    with pytest.raises(TypeError):
        torch.argsort(x, dim=(0, 1))


def test_sort_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]],
                     device="cuda", requires_grad=True)

    result = torch.sort(x, dim=1, descending=True)
    (result.values * torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                                  device="cuda")).sum().backward()

    assert result.values.device == "cuda"
    assert result.indices.device == "cuda"
    assert x.grad.device == "cuda"
    assert np.allclose(result.values.cpu().numpy(), [[4.0, 2.0, 1.0], [5.0, 3.0, 0.0]])
    assert np.allclose(result.indices.cpu().numpy(), [[1, 2, 0], [2, 0, 1]])
    assert torch.allclose(x.grad.cpu(), [
        [3.0, 1.0, 2.0],
        [5.0, 6.0, 4.0],
    ])
