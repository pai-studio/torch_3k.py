import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch_1k as torch
import torch_1k.nn as nn


def test_batchnorm1d_training_normalizes_and_backprops_affine():
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]],
        requires_grad=True,
    )
    bn = nn.BatchNorm1d(2)

    y = bn(x)
    loss = (y * y).sum()
    loss.backward()

    assert np.allclose(y.detach().numpy().mean(axis=0), [0.0, 0.0], atol=1e-6)
    assert np.allclose(y.detach().numpy().var(axis=0), [1.0, 1.0], atol=1e-4)
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None
    assert x.grad.shape == x.shape


def test_batchnorm1d_running_stats_update_with_unbiased_var():
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    bn = nn.BatchNorm1d(2, momentum=0.5, affine=False)

    bn(x)

    batch = x.numpy()
    expected_mean = 0.5 * batch.mean(axis=0)
    expected_var = 0.5 * np.ones(2) + 0.5 * batch.var(axis=0, ddof=1)
    assert np.allclose(bn.running_mean.numpy(), expected_mean)
    assert np.allclose(bn.running_var.numpy(), expected_var)


def test_batchnorm_eval_uses_running_stats():
    x = torch.tensor([[10.0, 20.0], [30.0, 50.0]])
    bn = nn.BatchNorm1d(2, affine=False)
    bn.running_mean.data[...] = np.array([1.0, 2.0])
    bn.running_var.data[...] = np.array([4.0, 9.0])
    bn.eval()

    y = bn(x)

    expected = (x.numpy() - np.array([1.0, 2.0])) / np.sqrt(np.array([4.0, 9.0]) + bn.eps)
    assert np.allclose(y.numpy(), expected)


def test_batchnorm2d_normalizes_per_channel():
    x_np = np.arange(24).reshape(2, 3, 2, 2).astype("float32")
    x = torch.tensor(x_np, requires_grad=True)
    bn = nn.BatchNorm2d(3, affine=False)

    y = bn(x)
    y.sum().backward()

    assert y.shape == x.shape
    assert np.allclose(y.detach().numpy().mean(axis=(0, 2, 3)), [0.0, 0.0, 0.0], atol=1e-6)
    assert x.grad.shape == x.shape


def test_batchnorm_buffers_are_in_state_dict_and_loadable():
    bn = nn.BatchNorm1d(2)
    bn(torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]]))

    state = bn.state_dict()
    restored = nn.BatchNorm1d(2)
    restored.load_state_dict(state)

    assert list(state.keys()) == ["weight", "bias", "running_mean", "running_var"]
    assert np.allclose(restored.running_mean.numpy(), bn.running_mean.numpy())
    assert np.allclose(restored.running_var.numpy(), bn.running_var.numpy())
    assert list(name for name, _ in bn.named_parameters()) == ["weight", "bias"]
    assert list(name for name, _ in bn.named_buffers()) == ["running_mean", "running_var"]


def test_batchnorm_nested_state_dict_contains_buffers():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 1))

    state = model.state_dict()

    assert "0.running_mean" in state
    assert "0.running_var" in state
    assert "0.weight" in state
    assert "1.weight" in state


def test_batchnorm_rejects_bad_input_dim():
    with pytest.raises(ValueError):
        nn.BatchNorm1d(2)(torch.zeros(2, 2, 2, 2))
    with pytest.raises(ValueError):
        nn.BatchNorm2d(2)(torch.zeros(2, 2))


def test_batchnorm_cuda_buffers_if_available():
    if not torch.cuda.is_available():
        return

    bn = nn.BatchNorm2d(2).to("cuda")
    x = torch.tensor(np.random.randn(2, 2, 3, 3).astype("float32"),
                     device="cuda", requires_grad=True)
    y = bn(x)
    y.sum().backward()

    assert bn.weight.device == "cuda"
    assert bn.running_mean.device == "cuda"
    assert bn.running_var.device == "cuda"
    assert y.device == "cuda"
    assert x.grad.device == "cuda"
