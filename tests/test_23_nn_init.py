import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch_1k as torch
import torch_1k.nn as nn


def test_constant_zeros_ones_modify_in_place():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    returned = nn.init.constant_(x, 3.5)
    assert returned is x
    assert np.allclose(x.numpy(), [[3.5, 3.5], [3.5, 3.5]])

    nn.init.zeros_(x)
    assert np.allclose(x.numpy(), np.zeros((2, 2)))

    nn.init.ones_(x)
    assert np.allclose(x.numpy(), np.ones((2, 2)))
    assert x.requires_grad is True


def test_uniform_and_normal_return_same_tensor():
    torch.manual_seed(0)
    x = torch.zeros(100)

    returned = nn.init.uniform_(x, a=-0.25, b=0.25)
    assert returned is x
    assert np.all(x.numpy() >= -0.25)
    assert np.all(x.numpy() <= 0.25)

    before = x.numpy().copy()
    returned = nn.init.normal_(x, mean=2.0, std=0.01)
    assert returned is x
    assert not np.allclose(before, x.numpy())
    assert abs(x.numpy().mean() - 2.0) < 0.01


def test_xavier_uniform_uses_fan_metadata(monkeypatch):
    layer = nn.Linear(2, 8)
    calls = {}

    def fake_uniform(a, b, size):
        calls['a'] = a
        calls['b'] = b
        calls['size'] = size
        return np.zeros(size)

    monkeypatch.setattr(np.random, 'uniform', fake_uniform)

    nn.init.xavier_uniform_(layer.weight, gain=1.0)

    bound = math.sqrt(6.0 / (2 + 8))
    assert calls == {'a': -bound, 'b': bound, 'size': (2, 8)}
    assert np.allclose(layer.weight.numpy(), np.zeros((2, 8)))


def test_kaiming_uniform_uses_linear_fan_in_metadata(monkeypatch):
    layer = nn.Linear(2, 8)
    calls = {}

    def fake_uniform(a, b, size):
        calls['a'] = a
        calls['b'] = b
        calls['size'] = size
        return np.zeros(size)

    monkeypatch.setattr(np.random, 'uniform', fake_uniform)

    nn.init.kaiming_uniform_(
        layer.weight,
        a=0,
        mode='fan_in',
        nonlinearity='relu',
    )

    bound = math.sqrt(6.0 / 2)
    assert calls == {'a': -bound, 'b': bound, 'size': (2, 8)}


def test_kaiming_uniform_uses_conv_fan_metadata(monkeypatch):
    conv = nn.Conv2d(3, 5, kernel_size=3)
    calls = {}

    def fake_uniform(a, b, size):
        calls['a'] = a
        calls['b'] = b
        calls['size'] = size
        return np.zeros(size)

    monkeypatch.setattr(np.random, 'uniform', fake_uniform)

    nn.init.kaiming_uniform_(
        conv.weight,
        a=0,
        mode='fan_in',
        nonlinearity='relu',
    )

    bound = math.sqrt(6.0 / (3 * 3 * 3))
    assert calls == {'a': -bound, 'b': bound, 'size': (5, 3, 3, 3)}


def test_init_rejects_non_tensor():
    with pytest.raises(TypeError):
        nn.init.zeros_([1.0, 2.0])


def test_kaiming_uniform_rejects_bad_mode():
    x = torch.zeros(2, 3)

    with pytest.raises(ValueError):
        nn.init.kaiming_uniform_(x, mode='bad')


def test_cuda_init_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.zeros(2, 3, device="cuda")
    nn.init.ones_(x)
    assert x.device == "cuda"
    assert np.allclose(x.cpu().numpy(), np.ones((2, 3)))

    nn.init.uniform_(x, a=-1.0, b=1.0)
    assert x.device == "cuda"
    assert np.all(x.cpu().numpy() >= -1.0)
    assert np.all(x.cpu().numpy() <= 1.0)
