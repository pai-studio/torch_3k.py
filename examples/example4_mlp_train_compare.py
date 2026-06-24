"""PyTorch-compatible MLP classification example.

The training body is shared by PyTorch and torch_1k. Switch implementations
with USE_TORCH_1K=0 or USE_TORCH_1K=1.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
    import torch_1k.nn as nn
    import torch_1k.optim as optim
else:
    import torch
    import torch.nn as nn
    import torch.optim as optim


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


def make_dataset(device):
    x = torch.tensor([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]).to(device)
    y = torch.tensor([0, 1, 1, 0]).to(device)
    return x, y


def run(epochs=1200, lr=0.2):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x, y = make_dataset(device)
    model = XORMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == y).float().mean().item()

    return loss.item(), accuracy, pred.detach().cpu().numpy()


if __name__ == "__main__":
    final_loss, accuracy, pred = run()
    print(f"final_loss={final_loss:.6f}")
    print(f"accuracy={accuracy:.6f}")
    print(f"pred={pred}")
    assert accuracy == 1.0
