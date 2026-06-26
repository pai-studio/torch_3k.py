import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch as pytorch
import torch_1k as torch


def _assert_amax_matches_pytorch(data, dim=None, keepdim=False):
    x = torch.tensor(data, requires_grad=True)
    y = torch.amax(x, dim=dim, keepdim=keepdim)
    y.sum().backward()

    tx = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    ty = pytorch.amax(tx, dim=dim, keepdim=keepdim)
    ty.sum().backward()

    assert np.allclose(y.numpy(), ty.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())


def test_amax_global_splits_gradient_between_ties():
    data = np.array([1.0, 3.0, 3.0, 2.0])

    _assert_amax_matches_pytorch(data)


def test_amax_tuple_dim_matches_pytorch():
    data = np.array([
        [[1.0, 3.0], [3.0, 2.0]],
        [[4.0, 4.0], [1.0, 4.0]],
    ])

    _assert_amax_matches_pytorch(data, dim=(1, 2))


def test_amax_keepdim_and_negative_dim_match_pytorch():
    data = np.array([
        [[1.0, 3.0], [3.0, 2.0]],
        [[4.0, 4.0], [1.0, 4.0]],
    ])

    _assert_amax_matches_pytorch(data, dim=(1, 2), keepdim=True)
    _assert_amax_matches_pytorch(data, dim=(-1,), keepdim=True)


def test_tensor_amax_method_matches_top_level():
    data = np.array([
        [[1.0, 3.0], [3.0, 2.0]],
        [[4.0, 4.0], [1.0, 4.0]],
    ])
    x = torch.tensor(data)

    assert torch.allclose(x.amax(dim=(1, 2)), torch.amax(x, dim=(1, 2)))
    assert torch.allclose(x.amax(axis=(1, 2), keepdims=True),
                          torch.amax(x, axis=(1, 2), keepdims=True))


def test_torch_max_dim_path_still_returns_indices_and_argmax_gradient():
    x = torch.tensor([[1.0, 3.0, 3.0]], requires_grad=True)

    result = torch.max(x, dim=1)
    result.values.sum().backward()

    assert np.allclose(result.values.numpy(), [3.0])
    assert np.allclose(result.indices.numpy(), [1])
    assert np.allclose(x.grad.numpy(), [[0.0, 1.0, 0.0]])


def test_amax_rejects_bad_dims():
    x = torch.tensor(np.ones((2, 3)))

    with pytest.raises(ValueError):
        torch.amax(x, dim=(1, 1))
    with pytest.raises(IndexError):
        torch.amax(x, dim=3)
    with pytest.raises(TypeError):
        torch.amax(x, dim="bad")


def test_amax_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor([
        [[1.0, 3.0], [3.0, 2.0]],
        [[4.0, 4.0], [1.0, 4.0]],
    ], device="cuda", requires_grad=True)

    y = torch.amax(x, dim=(1, 2), keepdim=True)
    y.sum().backward()

    assert y.device == "cuda"
    assert y.shape == (2, 1, 1)
    assert x.grad.device == "cuda"
    assert np.allclose(y.cpu().numpy(), [[[3.0]], [[4.0]]])
