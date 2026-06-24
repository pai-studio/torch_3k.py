import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch_1k as torch


def test_global_argmax_returns_flat_index():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 2.0]])

    y = torch.argmax(x)

    assert y.item() == 3
    assert y.shape == ()
    assert y.requires_grad is False


def test_dim_argmax_and_tensor_method():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 2.0]])

    top_level = torch.argmax(x, dim=1)
    method = x.argmax(dim=1)

    assert np.allclose(top_level.numpy(), [1, 0])
    assert np.allclose(method.numpy(), [1, 0])
    assert top_level.requires_grad is False
    assert method.requires_grad is False


def test_argmax_keepdim():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 2.0]])

    y = x.argmax(dim=1, keepdim=True)

    assert y.shape == (2, 1)
    assert np.allclose(y.numpy(), [[1], [0]])


def test_argmax_axis_and_keepdims_aliases():
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, 2.0]])

    y = torch.argmax(x, axis=-1, keepdims=True)

    assert y.shape == (2, 1)
    assert np.allclose(y.numpy(), [[1], [0]])


def test_argmax_accepts_plain_input():
    y = torch.argmax([[1.0, 3.0], [5.0, 4.0]], dim=0)

    assert np.allclose(y.numpy(), [1, 1])
    assert y.requires_grad is False


def test_argmax_rejects_tuple_dim():
    x = torch.tensor([[1.0, 3.0], [5.0, 4.0]])

    with pytest.raises(TypeError):
        x.argmax(dim=(0, 1))


def test_cuda_argmax_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor([[1.0, 3.0], [5.0, 4.0]], device="cuda")
    y = x.argmax(dim=1, keepdim=True)

    assert y.device == "cuda"
    assert y.shape == (2, 1)
    assert np.allclose(y.cpu().numpy(), [[1], [0]])
