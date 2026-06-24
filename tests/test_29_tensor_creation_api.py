import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch


def test_arange_single_end_start_end_and_step():
    assert np.allclose(torch.arange(5).numpy(), [0, 1, 2, 3, 4])
    assert np.allclose(torch.arange(1, 6, 2).numpy(), [1, 3, 5])
    assert torch.arange(3).requires_grad is False


def test_zeros_like_and_ones_like_inherit_shape_and_dtype():
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    assert zeros.shape == x.shape
    assert ones.shape == x.shape
    assert zeros.dtype == x.dtype
    assert ones.dtype == x.dtype
    assert np.allclose(zeros.numpy(), [[0, 0], [0, 0]])
    assert np.allclose(ones.numpy(), [[1, 1], [1, 1]])
    assert zeros.requires_grad is False


def test_full_and_full_like():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    y = torch.full((2, 3), 7, dtype=torch.long)
    z = torch.full_like(x, 2.5)

    assert y.shape == (2, 3)
    assert y.dtype == np.int64
    assert np.allclose(y.numpy(), np.full((2, 3), 7))
    assert z.shape == x.shape
    assert z.dtype == x.dtype
    assert np.allclose(z.numpy(), np.full((2, 2), 2.5))


def test_randint_pytorch_signatures_and_dtype():
    torch.manual_seed(0)
    one_arg = torch.randint(10, (2, 3))
    torch.manual_seed(0)
    two_arg = torch.randint(0, 10, (2, 3), dtype=torch.long)

    assert one_arg.shape == (2, 3)
    assert two_arg.shape == (2, 3)
    assert one_arg.dtype == np.int64
    assert two_arg.dtype == np.int64
    assert np.allclose(one_arg.numpy(), two_arg.numpy())
    assert np.all((one_arg.numpy() >= 0) & (one_arg.numpy() < 10))
    assert one_arg.requires_grad is False


def test_creation_cuda_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.arange(4, device="cuda")
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    full = torch.full((2, 2), 3.0, device="cuda")
    full_like = torch.full_like(full, 5.0)
    randint = torch.randint(0, 5, (2, 2), device="cuda")

    assert x.device == "cuda"
    assert zeros.device == "cuda"
    assert ones.device == "cuda"
    assert full.device == "cuda"
    assert full_like.device == "cuda"
    assert randint.device == "cuda"
    assert np.allclose(x.cpu().numpy(), [0, 1, 2, 3])
    assert np.allclose(full_like.cpu().numpy(), np.full((2, 2), 5.0))
