import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch as pytorch
import torch_1k as torch


def _assert_amin_matches_pytorch(data, dim=None, keepdim=False):
    x = torch.tensor(data, requires_grad=True)
    y = torch.amin(x, dim=dim, keepdim=keepdim)
    y.sum().backward()

    tx = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    ty = pytorch.amin(tx, dim=dim, keepdim=keepdim)
    ty.sum().backward()

    assert np.allclose(y.numpy(), ty.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())


def test_amin_global_splits_gradient_between_ties():
    data = np.array([1.0, -2.0, -2.0, 3.0])

    _assert_amin_matches_pytorch(data)


def test_amin_tuple_dim_matches_pytorch():
    data = np.array([
        [[1.0, -3.0], [-3.0, 2.0]],
        [[-4.0, -4.0], [1.0, -4.0]],
    ])

    _assert_amin_matches_pytorch(data, dim=(1, 2))


def test_amin_keepdim_and_negative_dim_match_pytorch():
    data = np.array([
        [[1.0, -3.0], [-3.0, 2.0]],
        [[-4.0, -4.0], [1.0, -4.0]],
    ])

    _assert_amin_matches_pytorch(data, dim=(1, 2), keepdim=True)
    _assert_amin_matches_pytorch(data, dim=(-1,), keepdim=True)


def test_tensor_amin_method_matches_top_level():
    data = np.array([
        [[1.0, -3.0], [-3.0, 2.0]],
        [[-4.0, -4.0], [1.0, -4.0]],
    ])
    x = torch.tensor(data)

    assert torch.allclose(x.amin(dim=(1, 2)), torch.amin(x, dim=(1, 2)))
    assert torch.allclose(x.amin(axis=(1, 2), keepdims=True),
                          torch.amin(x, axis=(1, 2), keepdims=True))


def test_amin_rejects_bad_dims():
    x = torch.tensor(np.ones((2, 3)))

    with pytest.raises(ValueError):
        torch.amin(x, dim=(1, 1))
    with pytest.raises(IndexError):
        torch.amin(x, dim=3)
    with pytest.raises(TypeError):
        torch.amin(x, dim="bad")


def test_aminmax_global_matches_pytorch_return_fields():
    data = np.array([[3.0, -1.0, 2.0], [-4.0, 5.0, 5.0]])
    x = torch.tensor(data)
    result = torch.aminmax(x)

    tx = pytorch.tensor(data, dtype=pytorch.float64)
    expected = pytorch.aminmax(tx)

    assert hasattr(result, "min")
    assert hasattr(result, "max")
    assert np.allclose(result.min.numpy(), expected.min.numpy())
    assert np.allclose(result.max.numpy(), expected.max.numpy())


def test_aminmax_dim_keepdim_matches_pytorch():
    data = np.array([
        [[3.0, -1.0, 2.0], [-4.0, 5.0, 5.0]],
        [[6.0, 0.0, -2.0], [1.0, -2.0, 7.0]],
    ])
    x = torch.tensor(data)
    result = torch.aminmax(x, dim=1, keepdim=True)

    tx = pytorch.tensor(data, dtype=pytorch.float64)
    expected = pytorch.aminmax(tx, dim=1, keepdim=True)

    assert np.allclose(result.min.numpy(), expected.min.numpy())
    assert np.allclose(result.max.numpy(), expected.max.numpy())
    assert torch.allclose(x.aminmax(axis=1, keepdims=True).min, result.min)


def test_aminmax_rejects_tuple_dim_like_pytorch():
    x = torch.tensor(np.ones((2, 3, 4)))

    with pytest.raises(TypeError):
        torch.aminmax(x, dim=(1, 2))


def test_aminmax_backward_reports_unimplemented_derivative():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    result = torch.aminmax(x)

    with pytest.raises(RuntimeError, match="aminmax"):
        (result.min + result.max).backward()


def test_amin_and_aminmax_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor([
        [[1.0, -3.0], [-3.0, 2.0]],
        [[-4.0, -4.0], [1.0, -4.0]],
    ], device="cuda", requires_grad=True)

    y = torch.amin(x, dim=(1, 2), keepdim=True)
    y.sum().backward()
    result = torch.aminmax(x.detach(), dim=1, keepdim=True)

    assert y.device == "cuda"
    assert y.shape == (2, 1, 1)
    assert x.grad.device == "cuda"
    assert result.min.device == "cuda"
    assert result.max.device == "cuda"
    assert np.allclose(y.cpu().numpy(), [[[-3.0]], [[-4.0]]])
