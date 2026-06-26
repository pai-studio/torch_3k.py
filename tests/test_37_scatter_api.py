import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch as pytorch
import torch_1k as torch


def _assert_scatter_matches_pytorch(op_name, input_data, index_data, src_data,
                                   dim):
    x = torch.tensor(input_data, requires_grad=True)
    index = torch.tensor(index_data)
    src = torch.tensor(src_data, requires_grad=True)
    y = getattr(torch, op_name)(x, dim, index, src)
    (y * y).sum().backward()

    tx = pytorch.tensor(input_data, dtype=pytorch.float64,
                       requires_grad=True)
    tindex = pytorch.tensor(index_data, dtype=pytorch.int64)
    tsrc = pytorch.tensor(src_data, dtype=pytorch.float64,
                         requires_grad=True)
    ty = getattr(pytorch, op_name)(tx, dim, tindex, tsrc)
    (ty * ty).sum().backward()

    assert np.allclose(y.numpy(), ty.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())
    assert np.allclose(src.grad.numpy(), tsrc.grad.detach().numpy())


def test_scatter_add_repeated_indices_match_pytorch():
    input_data = np.zeros((2, 4), dtype=np.float64)
    index_data = np.array([[0, 1, 1], [2, 0, 3]], dtype=np.int64)
    src_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    _assert_scatter_matches_pytorch(
        "scatter_add", input_data, index_data, src_data, dim=1,
    )


def test_scatter_without_repeated_indices_match_pytorch():
    input_data = np.zeros((2, 4), dtype=np.float64)
    index_data = np.array([[0, 2, 3], [1, 0, 2]], dtype=np.int64)
    src_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    _assert_scatter_matches_pytorch(
        "scatter", input_data, index_data, src_data, dim=1,
    )


def test_tensor_scatter_methods_match_top_level_functions():
    input_data = np.zeros((2, 4), dtype=np.float64)
    index = torch.tensor(np.array([[0, 2, 3], [1, 0, 2]], dtype=np.int64))
    src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = torch.tensor(input_data)

    assert torch.allclose(x.scatter(1, index, src),
                          torch.scatter(x, 1, index, src))
    assert torch.allclose(x.scatter_add(1, index, src),
                          torch.scatter_add(x, 1, index, src))


def test_scatter_add_dim0_and_negative_dim_match_pytorch():
    input_data = np.zeros((3, 2, 4), dtype=np.float64)
    index_data = np.array([
        [[0, 1, 2, 0], [1, 2, 0, 1]],
        [[2, 0, 1, 2], [0, 1, 2, 0]],
    ], dtype=np.int64)
    src_data = np.arange(16, dtype=np.float64).reshape(2, 2, 4) / 5.0

    _assert_scatter_matches_pytorch(
        "scatter_add", input_data, index_data, src_data, dim=0,
    )

    input_data = np.zeros((2, 3, 4), dtype=np.float64)
    index_data = np.array([
        [[0, 1], [2, 3], [1, 0]],
        [[3, 2], [0, 1], [2, 3]],
    ], dtype=np.int64)
    src_data = np.arange(12, dtype=np.float64).reshape(2, 3, 2) / 7.0
    _assert_scatter_matches_pytorch(
        "scatter_add", input_data, index_data, src_data, dim=-1,
    )


def test_scatter_rejects_invalid_inputs():
    x = torch.tensor(np.zeros((2, 3), dtype=np.float64))

    with pytest.raises(ValueError):
        torch.scatter(x, 1, torch.tensor([0, 1]), torch.tensor([1.0, 2.0]))

    with pytest.raises(ValueError):
        torch.scatter(
            x,
            1,
            torch.tensor([[0, 1], [1, 2]]),
            torch.tensor([[1.0], [2.0]]),
        )

    with pytest.raises(IndexError):
        torch.scatter_add(
            x,
            1,
            torch.tensor([[0, 3], [1, 2]]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        )


def test_scatter_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor(np.zeros((2, 4), dtype=np.float64), device="cuda",
                     requires_grad=True)
    index = torch.tensor([[0, 1, 1], [2, 0, 3]], device="cuda")
    src = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                       device="cuda", requires_grad=True)

    y = torch.scatter_add(x, 1, index, src)
    y.sum().backward()

    assert y.device == "cuda"
    assert x.grad.device == "cuda"
    assert src.grad.device == "cuda"
    assert np.allclose(y.cpu().numpy(), [[1.0, 5.0, 0.0, 0.0],
                                         [5.0, 0.0, 4.0, 6.0]])
