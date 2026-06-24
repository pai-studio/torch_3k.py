import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
import torch_1k.nn as nn
import torch_1k.optim as optim


def _train_steps(model, optimizer, x, y, steps):
    criterion = nn.CrossEntropyLoss()
    for _ in range(steps):
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_adam_state_dict_resume_matches_uninterrupted_training():
    torch.manual_seed(0)
    x = torch.tensor([[0.2, 0.8], [0.7, 0.1], [0.4, 0.6]])
    y = torch.tensor([1, 0, 1])
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    _train_steps(model, optimizer, x, y, steps=6)
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()

    _train_steps(model, optimizer, x, y, steps=4)
    expected = model(x).detach().numpy()

    restored = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    restored.load_state_dict(model_state)
    restored_optimizer = optim.Adam(restored.parameters(), lr=0.001)
    restored_optimizer.load_state_dict(optimizer_state)

    assert restored_optimizer.lr == 0.05
    assert restored_optimizer.t == 6

    _train_steps(restored, restored_optimizer, x, y, steps=4)
    actual = restored(x).detach().numpy()

    assert restored_optimizer.t == 10
    assert np.allclose(expected, actual)


def test_sgd_state_dict_roundtrip_restores_lr():
    p = torch.tensor([1.0])
    optimizer = optim.SGD([p], lr=0.2)
    state = optimizer.state_dict()

    restored = optim.SGD([p], lr=0.01)
    restored.load_state_dict(state)

    assert restored.lr == 0.2
    assert state["state"] == {}


def test_momentum_sgd_state_dict_roundtrip_restores_velocity():
    p = torch.tensor([1.0, -1.0])
    p.grad = torch.tensor([0.5, -0.25])
    optimizer = optim.MomentumSGD([p], lr=0.1, momentum=0.8)
    optimizer.step()
    state = optimizer.state_dict()

    restored_param = torch.tensor([1.0, -1.0])
    restored = optim.MomentumSGD([restored_param], lr=0.01, momentum=0.1)
    restored.load_state_dict(state)
    restored_key = id(restored_param)

    assert restored.lr == 0.1
    assert restored.momentum == 0.8
    assert np.allclose(restored.velocity[restored_key], [-0.05, 0.025])
