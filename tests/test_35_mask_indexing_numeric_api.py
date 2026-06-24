import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch as pytorch
import torch_1k as torch


def test_comparison_ops_create_non_grad_bool_masks():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    assert np.allclose((x == 2.0).numpy(), [[False, True], [False, False]])
    assert np.allclose((x != 2.0).numpy(), [[True, False], [True, True]])
    assert np.allclose((x < 3.0).numpy(), [[True, True], [False, False]])
    assert np.allclose((x <= 3.0).numpy(), [[True, True], [True, False]])
    assert np.allclose((x > 2.0).numpy(), [[False, False], [True, True]])
    assert np.allclose((x >= 2.0).numpy(), [[False, True], [True, True]])
    assert (x > 2.0).requires_grad is False
    assert bool(torch.tensor(True)) is True
    with pytest.raises(ValueError):
        bool(x > 2.0)


def test_where_broadcast_forward_and_backward_match_pytorch():
    x_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    other_data = np.array([[10.0], [20.0]], dtype=np.float64)
    cond_data = np.array([[True, False, True], [False, True, False]])
    weights = np.arange(1, 7, dtype=np.float64).reshape(2, 3)

    x = torch.tensor(x_data, requires_grad=True)
    other = torch.tensor(other_data, requires_grad=True)
    result = torch.where(torch.tensor(cond_data), x, other)
    (result * torch.tensor(weights)).sum().backward()

    tx = pytorch.tensor(x_data, dtype=pytorch.float64, requires_grad=True)
    tother = pytorch.tensor(other_data, dtype=pytorch.float64, requires_grad=True)
    tresult = pytorch.where(
        pytorch.tensor(cond_data),
        tx,
        tother,
    )
    (tresult * pytorch.tensor(weights, dtype=pytorch.float64)).sum().backward()

    assert np.allclose(result.numpy(), tresult.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())
    assert np.allclose(other.grad.numpy(), tother.grad.detach().numpy())


def test_masked_fill_zeros_input_gradient_at_masked_positions():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                     requires_grad=True)
    mask = x > 3.0

    y = x.masked_fill(mask, -100.0)
    y.sum().backward()

    assert np.allclose(y.numpy(), [[1.0, 2.0, 3.0], [-100.0, -100.0, -100.0]])
    assert torch.allclose(x.grad, [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])


def test_index_select_repeated_indices_match_pytorch():
    data = np.arange(6, dtype=np.float64).reshape(2, 3)
    index_data = np.array([2, 0, 2], dtype=np.int64)
    weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    x = torch.tensor(data, requires_grad=True)
    y = torch.index_select(x, 1, torch.tensor(index_data))
    (y * torch.tensor(weights)).sum().backward()

    tx = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    ty = pytorch.index_select(tx, 1, pytorch.tensor(index_data))
    (ty * pytorch.tensor(weights, dtype=pytorch.float64)).sum().backward()

    assert np.allclose(y.numpy(), ty.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())


def test_gather_repeated_indices_match_pytorch():
    data = np.arange(6, dtype=np.float64).reshape(2, 3)
    index_data = np.array([[2, 2, 1], [0, 2, 2]], dtype=np.int64)
    weights = np.arange(1, 7, dtype=np.float64).reshape(2, 3)

    x = torch.tensor(data, requires_grad=True)
    y = x.gather(1, torch.tensor(index_data))
    (y * torch.tensor(weights)).sum().backward()

    tx = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    ty = pytorch.gather(tx, 1, pytorch.tensor(index_data))
    (ty * pytorch.tensor(weights, dtype=pytorch.float64)).sum().backward()

    assert np.allclose(y.numpy(), ty.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())


def test_abs_sqrt_clamp_clip_forward_and_backward_match_pytorch():
    x_data = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    pos_data = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64)

    x = torch.tensor(x_data, requires_grad=True)
    pos = torch.tensor(pos_data, requires_grad=True)
    y = torch.abs(x) + x.clamp(min=0.0, max=1.0) + torch.sqrt(pos)
    y.sum().backward()

    tx = pytorch.tensor(x_data, dtype=pytorch.float64, requires_grad=True)
    tpos = pytorch.tensor(pos_data, dtype=pytorch.float64, requires_grad=True)
    ty = pytorch.abs(tx) + tx.clamp(min=0.0, max=1.0) + pytorch.sqrt(tpos)
    ty.sum().backward()

    assert np.allclose(y.numpy(), ty.detach().numpy())
    assert np.allclose(x.grad.numpy(), tx.grad.detach().numpy())
    assert np.allclose(pos.grad.numpy(), tpos.grad.detach().numpy())
    assert np.allclose(torch.clip(x, min=-0.5, max=1.5).numpy(),
                       pytorch.clip(tx.detach(), min=-0.5, max=1.5).numpy())


def test_gather_rejects_bad_index_rank():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError):
        torch.gather(x, 1, torch.tensor([0, 1]))


def test_mask_indexing_numeric_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]],
                     device="cuda", requires_grad=True)
    index = torch.tensor([[1, 2], [0, 2]], device="cuda")
    gathered = torch.gather(x, 1, index)
    masked = x.masked_fill(x < 2.0, 0.0)
    y = gathered.sum() + masked.clamp(max=4.0).sum()
    y.backward()

    assert gathered.device == "cuda"
    assert masked.device == "cuda"
    assert x.grad.device == "cuda"
    assert np.allclose(gathered.cpu().numpy(), [[4.0, 2.0], [3.0, 5.0]])
