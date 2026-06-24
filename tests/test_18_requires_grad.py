import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
import torch_1k.nn as nn
from torch_1k import Tensor, allclose
from torch_1k.utils.data import DataLoader, TensorDataset


def test_torch_tensor_defaults_to_no_grad_but_can_request_grad():
    x = torch.tensor([1.0, 2.0])
    y = (x * x).sum()
    y.backward()

    assert x.requires_grad is False
    assert y.requires_grad is False
    assert x.grad is None

    z = torch.tensor([1.0, 2.0], requires_grad=True)
    out = (z * z).sum()
    out.backward()

    assert z.requires_grad is True
    assert out.requires_grad is True
    assert allclose(z.grad, [2.0, 4.0])


def test_parameter_defaults_to_requires_grad_and_data_input_does_not():
    model = nn.Linear(2, 1)
    x = torch.tensor([[1.0, 2.0]])
    y = model(x).sum()
    y.backward()

    assert x.requires_grad is False
    assert x.grad is None
    assert model.weight.requires_grad is True
    assert model.weight.grad is not None


def test_no_grad_overrides_explicit_requires_grad():
    x = torch.tensor([1.0, 2.0], requires_grad=True)

    with torch.no_grad():
        y = (x * x).sum()

    assert y.requires_grad is False
    assert y.creator is None


def test_requires_grad_inplace_to_detach_and_dtype_conversion():
    x = torch.tensor([1.0, 2.0])
    x.requires_grad_()

    assert x.requires_grad is True
    assert x.to("cpu").requires_grad is True
    assert x.float().requires_grad is True
    assert x.detach().requires_grad is False
    assert x.long().requires_grad is False


def test_direct_tensor_constructor_keeps_internal_autograd_default():
    x = Tensor(3.0)
    y = x * x
    y.backward()

    assert x.requires_grad is True
    assert np.allclose(x.grad.numpy(), 6.0)


def test_dataloader_batches_and_state_dict_are_plain_data_by_default():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([0, 1])
    batch_x, batch_y = next(iter(DataLoader(TensorDataset(x, y), batch_size=2)))
    model = nn.Linear(2, 1)
    state = model.state_dict()

    assert batch_x.requires_grad is False
    assert batch_y.requires_grad is False
    assert all(value.requires_grad is False for value in state.values())
