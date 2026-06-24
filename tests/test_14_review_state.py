import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
import torch_1k.nn as nn
import torch_1k.optim as optim


def test_eval_does_not_disable_autograd():
    model = nn.Linear(2, 1)
    model.eval()
    x = torch.tensor([[1.0, 2.0]])
    y = model(x).sum()
    y.backward()

    assert model.training is False
    assert model.weight.grad is not None


def test_train_eval_recursive_mode():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1))
    model.eval()

    assert model.training is False
    assert model.__dict__["0"].training is False
    assert model.__dict__["1"].training is False

    model.train()
    assert model.training is True
    assert model.__dict__["0"].training is True


def test_dtype_constants_exported():
    x = torch.tensor([1, 2, 3], dtype=torch.long)
    y = torch.tensor([1.0, 2.0], dtype=torch.float32)

    assert x.dtype == np.int64
    assert y.dtype == np.float32


def test_state_dict_roundtrip_and_save_load(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.tensor([[0.2, 0.8], [0.7, 0.1]])
    y = torch.tensor([1, 0])
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for _ in range(10):
        loss = nn.CrossEntropyLoss()(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    expected = model(x).detach().numpy()
    state = model.state_dict()
    path = tmp_path / "model.pkl"
    torch.save(state, path)
    loaded = torch.load(path)

    restored = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    restored.load_state_dict(loaded)
    actual = restored(x).detach().numpy()
    loaded["0.weight"].data.fill(0)
    actual_after_state_mutation = restored(x).detach().numpy()

    assert list(state.keys()) == ["0.weight", "0.bias", "2.weight", "2.bias"]
    assert np.allclose(expected, actual)
    assert np.allclose(actual, actual_after_state_mutation)

    loaded["unexpected"] = torch.tensor([1.0])
    try:
        restored.load_state_dict(loaded)
    except KeyError:
        pass
    else:
        assert False, "strict load_state_dict should reject unexpected keys"
