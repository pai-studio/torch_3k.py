import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
from torch_1k import allclose


def test_pytorch_compatible_device_api_cpu():
    x = torch.tensor([1.0, 2.0, 3.0]).to("cpu")
    y = (x * 2.0 + 1.0).detach().cpu().numpy()

    assert x.device == "cpu"
    assert isinstance(torch.cuda.is_available(), bool)
    assert np.allclose(y, [3.0, 5.0, 7.0])


def test_broadcast_grad_to_scalar():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)

    y = ((x + b) * b).sum()
    y.backward()

    assert allclose(x.grad, [[2.0, 2.0], [2.0, 2.0]])
    assert allclose(b.grad, 26.0)


def test_cuda_tensor_basic_if_available():
    if not torch.cuda.is_available():
        return

    x = torch.tensor([1.0, 2.0, 3.0], device="cuda", requires_grad=True)
    y = (x * x).sum()
    y.backward()

    assert x.device == "cuda"
    assert y.device == "cuda"
    assert x.grad.device == "cuda"
    assert allclose(x.grad.cpu(), [2.0, 4.0, 6.0])
