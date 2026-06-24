import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
import torch_1k.nn as nn
from torch_1k import allclose


def test_softmax_tensor_and_functional_api_normalize_rows():
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, -1.0, 0.5]], requires_grad=True)

    tensor_probs = x.softmax(dim=1)
    functional_probs = nn.functional.softmax(x, dim=1)

    assert np.allclose(tensor_probs.numpy().sum(axis=1), [1.0, 1.0])
    assert np.allclose(tensor_probs.numpy(), functional_probs.numpy())


def test_log_softmax_matches_softmax_and_backprops():
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, -1.0, 0.5]], requires_grad=True)

    probs = torch.softmax(x, dim=1)
    log_probs = torch.log_softmax(x, dim=1)
    loss = -log_probs[0, 2] - log_probs[1, 0]
    loss.backward()

    assert np.allclose(log_probs.exp().numpy(), probs.numpy())
    assert x.grad.shape == x.shape
    assert x.grad is not None


def test_softmax_and_logsoftmax_modules_match_functional_api():
    x = torch.tensor([[0.2, 0.8], [2.0, -1.0]], requires_grad=True)

    assert allclose(nn.Softmax(dim=1)(x), nn.functional.softmax(x, dim=1))
    assert allclose(nn.LogSoftmax(dim=1)(x), nn.functional.log_softmax(x, dim=1))
