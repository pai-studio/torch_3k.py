import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch as pytorch
import torch_1k as torch


def _array(shape, offset=0.0):
    size = int(np.prod(shape))
    return (np.arange(size, dtype=np.float64).reshape(shape) + offset) / 7.0


def _assert_einsum_matches_pytorch(equation, arrays):
    xs = [torch.tensor(array, requires_grad=True) for array in arrays]
    y = torch.einsum(equation, *xs)
    y.sum().backward()

    txs = [
        pytorch.tensor(array, dtype=pytorch.float64, requires_grad=True)
        for array in arrays
    ]
    ty = pytorch.einsum(equation, *txs)
    ty.sum().backward()

    assert np.allclose(y.numpy(), ty.detach().numpy())
    for x, tx in zip(xs, txs):
        assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())


def test_einsum_matmul_forward_and_backward():
    x_data = np.arange(6).reshape(2, 3).astype("float32")
    w_data = np.arange(12).reshape(3, 4).astype("float32")
    x = torch.tensor(x_data, requires_grad=True)
    w = torch.tensor(w_data, requires_grad=True)

    y = torch.einsum("ij,jk->ik", x, w)
    y.sum().backward()

    expected = x_data @ w_data
    expected_x_grad = np.ones((2, 4), dtype="float32") @ w_data.T
    expected_w_grad = x_data.T @ np.ones((2, 4), dtype="float32")
    assert np.allclose(y.numpy(), expected)
    assert torch.allclose(x.grad, expected_x_grad)
    assert torch.allclose(w.grad, expected_w_grad)


def test_einsum_implicit_output_matches_explicit_matmul():
    x = torch.tensor(np.arange(6).reshape(2, 3).astype("float32"))
    w = torch.tensor(np.arange(12).reshape(3, 4).astype("float32"))

    implicit = torch.einsum("ij,jk", x, w)
    explicit = torch.einsum("ij,jk->ik", x, w)

    assert torch.allclose(implicit, explicit)


def test_einsum_attention_score_backward_shapes():
    q = torch.tensor(np.random.randn(2, 3, 4).astype("float32"),
                     requires_grad=True)
    k = torch.tensor(np.random.randn(2, 5, 4).astype("float32"),
                     requires_grad=True)

    scores = torch.einsum("bqd,bkd->bqk", q, k)
    scores.sum().backward()

    assert scores.shape == (2, 3, 5)
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape


def test_einsum_single_input_reduction_backward():
    x = torch.tensor(np.arange(6).reshape(2, 3).astype("float32"),
                     requires_grad=True)

    y = torch.einsum("ij->i", x)
    y.sum().backward()

    assert np.allclose(y.numpy(), [3.0, 12.0])
    assert torch.allclose(x.grad, np.ones((2, 3), dtype="float32"))


def test_einsum_dot_product_scalar_backward():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

    z = torch.einsum("i,i->", x, y)
    z.backward()

    assert np.allclose(z.item(), 32.0)
    assert torch.allclose(x.grad, [4.0, 5.0, 6.0])
    assert torch.allclose(y.grad, [1.0, 2.0, 3.0])


@pytest.mark.parametrize("equation,arrays", [
    ("ij,jk->ik", [_array((2, 3), 1.0), _array((3, 4), 2.0)]),
    ("bij,bjk->bik", [_array((2, 3, 4), 1.0), _array((2, 4, 5), 2.0)]),
    ("i,i->", [_array((4,), 1.0), _array((4,), 2.0)]),
    ("i,j->ij", [_array((3,), 1.0), _array((4,), 2.0)]),
    ("ij->i", [_array((2, 3), 1.0)]),
    ("ij->", [_array((2, 3), 1.0)]),
    ("ij,ij->", [_array((2, 3), 1.0), _array((2, 3), 2.0)]),
    ("bqd,bkd->bqk", [_array((2, 3, 4), 1.0), _array((2, 5, 4), 2.0)]),
    ("bqk,bkd->bqd", [_array((2, 3, 5), 1.0), _array((2, 5, 4), 2.0)]),
    ("ij,j->ij", [_array((2, 1), 1.0), _array((3,), 2.0)]),
    ("ij,j->i", [_array((2, 1), 1.0), _array((3,), 2.0)]),
    ("ijk,j->i", [_array((2, 1, 4), 1.0), _array((3,), 2.0)]),
])
def test_einsum_classic_torch_usages_match_pytorch(equation, arrays):
    _assert_einsum_matches_pytorch(equation, arrays)


def test_einsum_rejects_unsupported_equations():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(NotImplementedError):
        torch.einsum("ii->i", x)
    with pytest.raises(NotImplementedError):
        torch.einsum("...ij->...i", x)


def test_einsum_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor(np.arange(6).reshape(2, 3).astype("float32"),
                     device="cuda", requires_grad=True)
    w = torch.tensor(np.arange(12).reshape(3, 4).astype("float32"),
                     device="cuda", requires_grad=True)

    y = torch.einsum("ij,jk->ik", x, w)
    y.sum().backward()

    assert y.device == "cuda"
    assert x.grad.device == "cuda"
    assert w.grad.device == "cuda"
    assert np.allclose(y.cpu().numpy(), x.cpu().numpy() @ w.cpu().numpy())
