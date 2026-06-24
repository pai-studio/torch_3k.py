import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
import torch_1k.optim as optim


def test_sgd_param_groups_use_different_lr():
    p1 = torch.tensor([1.0])
    p2 = torch.tensor([1.0])
    p1.grad = torch.tensor([1.0])
    p2.grad = torch.tensor([1.0])
    optimizer = optim.SGD([
        {'params': [p1], 'lr': 0.1},
        {'params': [p2], 'lr': 0.01},
    ], lr=0.001)

    optimizer.step()

    assert np.allclose(p1.numpy(), [0.9])
    assert np.allclose(p2.numpy(), [0.99])
    assert optimizer.lr == 0.1


def test_momentum_sgd_param_groups_use_group_momentum():
    p1 = torch.tensor([1.0])
    p2 = torch.tensor([1.0])
    p1.grad = torch.tensor([1.0])
    p2.grad = torch.tensor([1.0])
    optimizer = optim.MomentumSGD([
        {'params': [p1], 'lr': 0.1, 'momentum': 0.0},
        {'params': [p2], 'lr': 0.1, 'momentum': 0.9},
    ])

    optimizer.step()
    p1.grad = torch.tensor([1.0])
    p2.grad = torch.tensor([1.0])
    optimizer.step()

    assert np.allclose(p1.numpy(), [0.8])
    assert np.allclose(p2.numpy(), [0.71])
    assert optimizer.momentum == 0.0


def test_adam_param_groups_use_different_lr_and_weight_decay():
    p1 = torch.tensor([1.0])
    p2 = torch.tensor([1.0])
    p1.grad = torch.tensor([1.0])
    p2.grad = torch.tensor([1.0])
    optimizer = optim.Adam([
        {'params': [p1], 'lr': 0.1, 'weight_decay': 0.0},
        {'params': [p2], 'lr': 0.01, 'weight_decay': 0.1},
    ], betas=(0.0, 0.0), eps=0.0)

    optimizer.step()

    assert np.allclose(p1.numpy(), [0.9])
    assert np.allclose(p2.numpy(), [0.99])
    assert optimizer.lr == 0.1
    assert optimizer.weight_decay == 0.0


def test_adamw_param_groups_use_decoupled_weight_decay():
    p1 = torch.tensor([1.0])
    p2 = torch.tensor([1.0])
    p1.grad = torch.tensor([0.0])
    p2.grad = torch.tensor([0.0])
    optimizer = optim.AdamW([
        {'params': [p1], 'lr': 0.1, 'weight_decay': 0.0},
        {'params': [p2], 'lr': 0.1, 'weight_decay': 0.1},
    ], betas=(0.0, 0.0), eps=1.0)

    optimizer.step()

    assert np.allclose(p1.numpy(), [1.0])
    assert np.allclose(p2.numpy(), [0.99])
    assert optimizer.decoupled_weight_decay == 0.0


def test_param_group_state_dict_roundtrip_restores_groups_and_state():
    p1 = torch.tensor([1.0])
    p2 = torch.tensor([2.0])
    p1.grad = torch.tensor([0.5])
    p2.grad = torch.tensor([0.25])
    optimizer = optim.Adam([
        {'params': [p1], 'lr': 0.1, 'weight_decay': 0.0},
        {'params': [p2], 'lr': 0.01, 'weight_decay': 0.2},
    ])
    optimizer.step()
    state = optimizer.state_dict()

    r1 = torch.tensor([1.0])
    r2 = torch.tensor([2.0])
    restored = optim.Adam([
        {'params': [r1], 'lr': 0.001},
        {'params': [r2], 'lr': 0.001},
    ])
    restored.load_state_dict(state)

    assert len(state['param_groups']) == 2
    assert state['param_groups'][0]['lr'] == 0.1
    assert state['param_groups'][1]['weight_decay'] == 0.2
    assert restored.param_groups[0]['lr'] == 0.1
    assert restored.param_groups[1]['lr'] == 0.01
    assert restored.param_groups[1]['weight_decay'] == 0.2
    assert restored.t == 1
    assert np.allclose(restored.m[id(r1)], optimizer.m[id(p1)])
    assert np.allclose(restored.v[id(r2)], optimizer.v[id(p2)])


def test_load_state_dict_rejects_different_param_group_count():
    p = torch.tensor([1.0])
    optimizer = optim.SGD([{'params': [p], 'lr': 0.1}])
    state = optimizer.state_dict()
    restored = optim.SGD([
        {'params': [torch.tensor([1.0])], 'lr': 0.1},
        {'params': [torch.tensor([2.0])], 'lr': 0.01},
    ])

    try:
        restored.load_state_dict(state)
    except ValueError:
        pass
    else:
        assert False, "load_state_dict should reject different param group count"


def test_cuda_param_groups_if_available():
    if not torch.cuda.is_available():
        return

    p1 = torch.tensor([1.0], device="cuda")
    p2 = torch.tensor([1.0], device="cuda")
    p1.grad = torch.tensor([1.0], device="cuda")
    p2.grad = torch.tensor([1.0], device="cuda")
    optimizer = optim.SGD([
        {'params': [p1], 'lr': 0.1},
        {'params': [p2], 'lr': 0.01},
    ])

    optimizer.step()

    assert p1.device == "cuda"
    assert p2.device == "cuda"
    assert np.allclose(p1.cpu().numpy(), [0.9])
    assert np.allclose(p2.cpu().numpy(), [0.99])
