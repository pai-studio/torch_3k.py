import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch as pytorch
import torch_1k as torch


def _assert_nonzero_matches_pytorch(data):
    ours = torch.nonzero(torch.tensor(data))
    refs = pytorch.nonzero(pytorch.tensor(data))
    assert np.allclose(ours.numpy(), refs.numpy())
    assert ours.requires_grad is False

    ours_tuple = torch.nonzero(torch.tensor(data), as_tuple=True)
    refs_tuple = pytorch.nonzero(pytorch.tensor(data), as_tuple=True)
    assert len(ours_tuple) == len(refs_tuple)
    for ours_index, refs_index in zip(ours_tuple, refs_tuple):
        assert np.allclose(ours_index.numpy(), refs_index.numpy())
        assert ours_index.requires_grad is False


def test_nonzero_bool_mask_matches_pytorch():
    mask = np.array([[True, False, True], [False, True, False]])

    _assert_nonzero_matches_pytorch(mask)


def test_nonzero_numeric_input_and_tensor_method_match_pytorch():
    data = np.array([[0.0, 2.0, 0.0], [-1.0, 0.0, 3.0]])
    x = torch.tensor(data)

    _assert_nonzero_matches_pytorch(data)
    assert np.allclose(x.nonzero().numpy(),
                       pytorch.tensor(data).nonzero().numpy())


def test_where_condition_returns_nonzero_tuple():
    mask = np.array([[True, False, True], [False, True, False]])

    ours = torch.where(torch.tensor(mask))
    refs = pytorch.where(pytorch.tensor(mask))

    assert len(ours) == len(refs)
    for ours_index, refs_index in zip(ours, refs):
        assert np.allclose(ours_index.numpy(), refs_index.numpy())
        assert ours_index.requires_grad is False


def test_nonzero_scalar_matches_pytorch_shape():
    active = torch.nonzero(torch.tensor(5))
    inactive = torch.nonzero(torch.tensor(0))
    active_tuple = torch.nonzero(torch.tensor(5), as_tuple=True)
    inactive_tuple = torch.nonzero(torch.tensor(0), as_tuple=True)

    assert active.shape == pytorch.nonzero(pytorch.tensor(5)).shape
    assert inactive.shape == pytorch.nonzero(pytorch.tensor(0)).shape
    assert np.allclose(active.numpy(), pytorch.nonzero(pytorch.tensor(5)).numpy())
    assert np.allclose(inactive.numpy(),
                       pytorch.nonzero(pytorch.tensor(0)).numpy())
    assert np.allclose(active_tuple[0].numpy(), [0])
    assert inactive_tuple[0].shape == (0,)


def test_where_three_argument_path_still_differentiates():
    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    other = torch.tensor([[10.0], [20.0]], requires_grad=True)
    condition = torch.tensor([[True, False, True], [False, True, False]])
    weights = torch.tensor(np.arange(1, 7, dtype=np.float64).reshape(2, 3))

    y = torch.where(condition, x, other)
    (y * weights).sum().backward()

    tx = pytorch.tensor([[1.0, 2.0, 3.0]], dtype=pytorch.float64,
                        requires_grad=True)
    tother = pytorch.tensor([[10.0], [20.0]], dtype=pytorch.float64,
                            requires_grad=True)
    ty = pytorch.where(
        pytorch.tensor([[True, False, True], [False, True, False]]),
        tx,
        tother,
    )
    (ty * pytorch.tensor(np.arange(1, 7, dtype=np.float64).reshape(2, 3))).sum().backward()

    assert np.allclose(y.numpy(), ty.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())
    assert np.allclose(other.grad.numpy(), tother.grad.detach().numpy())


def test_where_rejects_partial_or_non_bool_condition():
    with pytest.raises(TypeError):
        torch.where(torch.tensor([True, False]), torch.tensor([1.0]))

    with pytest.raises(TypeError):
        torch.where(torch.tensor([0, 1]))


def test_nonzero_where_cuda_if_available():
    if not torch.cuda.is_available():
        return

    mask = torch.tensor([[True, False, True], [False, True, False]],
                        device="cuda")
    coords = torch.nonzero(mask)
    rows, cols = torch.where(mask)

    assert coords.device == "cuda"
    assert rows.device == "cuda"
    assert cols.device == "cuda"
    assert np.allclose(coords.cpu().numpy(), [[0, 0], [0, 2], [1, 1]])
    assert np.allclose(rows.cpu().numpy(), [0, 0, 1])
    assert np.allclose(cols.cpu().numpy(), [0, 2, 1])
