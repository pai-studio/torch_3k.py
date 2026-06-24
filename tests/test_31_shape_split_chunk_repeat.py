import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch


def test_repeat_forward_and_backward():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    y = x.repeat(2, 3)
    y.sum().backward()

    assert y.shape == (4, 6)
    assert np.allclose(y.numpy(), np.tile(x.detach().numpy(), (2, 3)))
    assert torch.allclose(x.grad, np.full((2, 2), 6.0))


def test_repeat_can_add_leading_dimensions():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    y = x.repeat(2, 1)
    y.sum().backward()

    assert y.shape == (2, 3)
    assert np.allclose(y.numpy(), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert torch.allclose(x.grad, [2.0, 2.0, 2.0])


def test_split_by_size_and_gradient_scatter():
    x = torch.tensor(np.arange(10).reshape(5, 2).astype("float32"),
                     requires_grad=True)

    a, b, c = x.split(2, dim=0)
    loss = a.sum() + b.sum() * 2.0 + c.sum() * 3.0
    loss.backward()

    assert a.shape == (2, 2)
    assert b.shape == (2, 2)
    assert c.shape == (1, 2)
    assert torch.allclose(x.grad, [
        [1.0, 1.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])


def test_split_by_sections_and_negative_dim():
    x = torch.tensor(np.arange(12).reshape(2, 6).astype("float32"),
                     requires_grad=True)

    left, middle, right = torch.split(x, [1, 3, 2], dim=-1)
    loss = left.sum() + middle.sum() * 2.0 + right.sum() * 3.0
    loss.backward()

    assert left.shape == (2, 1)
    assert middle.shape == (2, 3)
    assert right.shape == (2, 2)
    assert torch.allclose(x.grad, [
        [1.0, 2.0, 2.0, 2.0, 3.0, 3.0],
        [1.0, 2.0, 2.0, 2.0, 3.0, 3.0],
    ])


def test_chunk_uses_nearly_even_splits():
    x = torch.arange(5)

    parts = torch.chunk(x, 3, dim=0)

    assert [part.shape for part in parts] == [(2,), (2,), (1,)]
    assert np.allclose(parts[0].numpy(), [0, 1])
    assert np.allclose(parts[1].numpy(), [2, 3])
    assert np.allclose(parts[2].numpy(), [4])


def test_split_chunk_repeat_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda",
                     requires_grad=True)

    repeated = x.repeat(2, 1)
    first, second = repeated.chunk(2, dim=0)
    loss = first.sum() + second.sum()
    loss.backward()

    assert repeated.device == "cuda"
    assert first.device == "cuda"
    assert second.device == "cuda"
    assert x.grad.device == "cuda"
    assert torch.allclose(x.grad.cpu(), [[2.0, 2.0], [2.0, 2.0]])
