import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
from torch_1k import allclose
from torch_1k.utils.data import DataLoader, TensorDataset


def test_stack_backward_to_inputs():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])

    z = torch.stack((x, y), dim=0)
    loss = z.sum()
    loss.backward()

    assert z.shape == (2, 2)
    assert allclose(x.grad, [1.0, 1.0])
    assert allclose(y.grad, [1.0, 1.0])


def test_tensor_dataset_dataloader_collates_tensors():
    x = torch.tensor(np.arange(12).reshape(6, 2).astype("float32"))
    y = torch.tensor(np.arange(6)).long()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batches = list(loader)
    batch_x, batch_y = batches[0]
    last_x, last_y = batches[1]

    assert len(dataset) == 6
    assert len(loader) == 2
    assert batch_x.shape == (4, 2)
    assert batch_y.shape == (4,)
    assert last_x.shape == (2, 2)
    assert last_y.shape == (2,)
    assert np.allclose(batch_x.numpy(), [[0, 1], [2, 3], [4, 5], [6, 7]])
    assert np.allclose(batch_y.numpy(), [0, 1, 2, 3])


def test_dataloader_drop_last_and_repeated_iteration():
    x = torch.tensor(np.arange(10).reshape(5, 2).astype("float32"))
    y = torch.tensor(np.arange(5)).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False, drop_last=True)

    first_pass = list(loader)
    second_pass = list(loader)

    assert len(loader) == 2
    assert [batch_x.shape for batch_x, _ in first_pass] == [(2, 2), (2, 2)]
    assert np.allclose(first_pass[0][0].numpy(), second_pass[0][0].numpy())


def test_tensor_dataset_accepts_numpy_arrays():
    x = np.arange(6).reshape(3, 2).astype("float32")
    y = np.array([0, 1, 0])
    loader = DataLoader(TensorDataset(x, y), batch_size=2, shuffle=False)

    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (2, 2)
    assert batch_y.shape == (2,)
    assert np.allclose(batch_x.numpy(), [[0, 1], [2, 3]])


def test_torch_utils_data_namespace_matches_pytorch_style():
    dataset = torch.utils.data.TensorDataset(torch.tensor([[1.0], [2.0]]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)

    (batch_x,) = next(iter(loader))

    assert batch_x.shape == (2, 1)
