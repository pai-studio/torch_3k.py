import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
import torch_1k.nn as nn
from torch_1k import allclose


def test_log_top_level_and_tensor_method_backward():
    x = torch.tensor([0.5, 2.0, 4.0], requires_grad=True)

    y = torch.log(x).sum() + x.log().sum()
    y.backward()

    assert np.allclose(torch.log(x).numpy(), np.log([0.5, 2.0, 4.0]))
    assert allclose(x.grad, 2.0 / np.array([0.5, 2.0, 4.0]))


def test_relu_top_level_tensor_method_and_nn_functional_backward():
    x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)

    y = torch.relu(x).sum() + x.relu().sum() + nn.functional.relu(x).sum()
    y.backward()

    assert np.allclose(torch.relu(x).numpy(), [0.0, 0.0, 2.0])
    assert allclose(x.grad, [0.0, 0.0, 3.0])


def test_common_tensor_function_methods_forward():
    x = torch.tensor([-1.0, 0.0, 1.0])

    assert np.allclose(x.exp().numpy(), np.exp([-1.0, 0.0, 1.0]))
    assert np.allclose(x.tanh().numpy(), np.tanh([-1.0, 0.0, 1.0]))
    assert np.allclose(x.sigmoid().numpy(), 1.0 / (1.0 + np.exp(-np.array([-1.0, 0.0, 1.0]))))
