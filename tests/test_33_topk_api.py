import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch


def test_topk_largest_values_and_indices():
    x = torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]])

    result = torch.topk(x, 2, dim=1)

    assert np.allclose(result.values.numpy(), [[4.0, 2.0], [5.0, 3.0]])
    assert np.allclose(result.indices.numpy(), [[1, 2], [2, 0]])
    assert result.values.requires_grad is False
    assert result.indices.requires_grad is False


def test_topk_smallest_and_negative_dim():
    x = torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]])

    values, indices = x.topk(2, dim=-1, largest=False)

    assert np.allclose(values.numpy(), [[1.0, 2.0], [0.0, 3.0]])
    assert np.allclose(indices.numpy(), [[0, 2], [1, 0]])


def test_topk_default_dim_is_last_dimension():
    x = torch.tensor(np.array([
        [[1.0, 3.0], [4.0, 2.0]],
        [[0.0, 5.0], [6.0, 1.0]],
    ]))

    result = torch.topk(x, 1)

    assert result.values.shape == (2, 2, 1)
    assert np.allclose(result.values.numpy(), [[[3.0], [4.0]], [[5.0], [6.0]]])
    assert np.allclose(result.indices.numpy(), [[[1], [0]], [[1], [0]]])


def test_topk_sorted_false_returns_topk_set():
    x = torch.tensor([[1.0, 4.0, 2.0, 3.0]])

    result = torch.topk(x, 3, dim=1, sorted=False)

    assert result.values.shape == (1, 3)
    assert sorted(result.values.numpy()[0].tolist()) == [2.0, 3.0, 4.0]
    assert sorted(result.indices.numpy()[0].tolist()) == [1, 2, 3]


def test_topk_backward_scatters_to_selected_values():
    x = torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]],
                     requires_grad=True)

    values = torch.topk(x, 2, dim=1).values
    (values * torch.tensor([[1.0, 2.0], [3.0, 4.0]])).sum().backward()

    assert torch.allclose(x.grad, [
        [0.0, 1.0, 2.0],
        [4.0, 0.0, 3.0],
    ])


def test_topk_allows_k_zero():
    x = torch.tensor([[1.0, 2.0, 3.0]])

    result = torch.topk(x, 0, dim=1)

    assert result.values.shape == (1, 0)
    assert result.indices.shape == (1, 0)


def test_topk_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor([[1.0, 4.0, 2.0], [3.0, 0.0, 5.0]],
                     device="cuda", requires_grad=True)

    result = torch.topk(x, 2, dim=1)
    result.values.sum().backward()

    assert result.values.device == "cuda"
    assert result.indices.device == "cuda"
    assert x.grad.device == "cuda"
    assert np.allclose(result.values.cpu().numpy(), [[4.0, 2.0], [5.0, 3.0]])
    assert torch.allclose(x.grad.cpu(), [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ])
