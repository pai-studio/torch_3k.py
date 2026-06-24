import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
import torch_1k.nn as nn
from torch_1k import allclose


def test_dropout_training_masks_scales_and_backprops():
    torch.manual_seed(0)
    x = torch.ones(40, requires_grad=True)
    dropout = nn.Dropout(p=0.5)

    y = dropout(x)
    y.sum().backward()
    values = y.detach().numpy()

    assert y.requires_grad is True
    assert set(np.unique(values)).issubset({0.0, 2.0})
    assert (values == 0.0).any()
    assert (values == 2.0).any()
    assert allclose(x.grad, values)


def test_dropout_eval_is_identity_and_keeps_requires_grad():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    dropout = nn.Dropout(p=0.75)
    dropout.eval()

    y = dropout(x)

    assert y is x
    assert y.requires_grad is True
    assert np.allclose(y.numpy(), x.numpy())


def test_functional_dropout_training_flag_and_edge_probabilities():
    x = torch.ones(4, requires_grad=True)

    assert nn.functional.dropout(x, p=0.5, training=False) is x
    assert nn.functional.dropout(x, p=0.0, training=True) is x
    assert np.allclose(nn.functional.dropout(x, p=1.0, training=True).numpy(), 0.0)


def test_dropout_rejects_invalid_probability():
    for p in (-0.1, 1.1):
        try:
            nn.Dropout(p=p)
        except ValueError:
            pass
        else:
            assert False, "Dropout should reject p outside [0, 1]"
