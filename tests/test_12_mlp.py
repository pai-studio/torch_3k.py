import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import torch_1k as torch
import torch_1k.nn as nn
import torch_1k.optim as optim


class XORMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


def test_cross_entropy_backward_shape():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    target = torch.tensor([0, 1])
    loss = nn.CrossEntropyLoss()(logits, target)

    loss.backward()

    assert loss.item() < 0.2
    assert logits.grad.shape == logits.shape


def test_mlp_xor_training():
    torch.manual_seed(0)
    x = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ])
    y = torch.tensor([0, 1, 1, 0])

    model = XORMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.2)

    for _ in range(1200):
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred = torch.argmax(model(x), dim=1)
        accuracy = (pred == y).float().mean().item()

    assert accuracy == 1.0
