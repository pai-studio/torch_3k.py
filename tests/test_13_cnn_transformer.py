import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import torch_1k as torch
import torch_1k.nn as nn
import torch_1k.optim as optim
from torch_1k import Tensor


def test_common_optimizers_step():
    for optimizer_cls in (optim.MomentumSGD, optim.Adam, optim.AdamW):
        p = Tensor([1.0, -1.0])
        p.grad = Tensor([0.5, -0.25])
        optimizer = optimizer_cls([p], lr=0.1)
        before = p.numpy().copy()
        optimizer.step()
        assert not np.allclose(before, p.numpy())


def test_cnn_modules_backward():
    torch.manual_seed(0)
    x = torch.tensor(np.random.randn(2, 1, 8, 8).astype("float32"))
    y = torch.tensor([0, 1])
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(2 * 2 * 2, 2),
    )
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()

    conv = model.__dict__["0"]
    assert conv.weight.grad.shape == conv.weight.shape


def test_transformer_encoder_layer_backward():
    torch.manual_seed(0)
    x = torch.tensor([[0, 2, 3], [1, 3, 4]]).long()
    y = torch.tensor([0, 1])
    embedding = nn.Embedding(8, 4)
    encoder = nn.TransformerEncoderLayer(
        d_model=4,
        nhead=1,
        dim_feedforward=8,
        batch_first=True,
    )
    classifier = nn.Linear(4, 2)

    logits = classifier(encoder(embedding(x))[:, 0, :])
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    assert embedding.weight.grad.shape == embedding.weight.shape
