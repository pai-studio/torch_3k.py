"""PyTorch-compatible CNN classification example on MNIST-like images.

The dataset is generated locally as 28x28 digit-like patterns, so the example
has no download dependency. Replace make_dataset() with a real MNIST loader to
train on MNIST without changing the model or training loop.
"""

import os
import sys
from pathlib import Path

import numpy as np

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


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(4 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


def make_dataset(device):
    rng = np.random.RandomState(0)
    images = []
    labels = []
    positions = [
        (2, 2), (2, 10), (2, 18), (10, 2), (10, 10),
        (10, 18), (18, 2), (18, 10), (18, 18), (5, 5),
    ]
    for label, (r, c) in enumerate(positions):
        for _ in range(2):
            image = rng.normal(0, 0.03, size=(28, 28)).astype("float32")
            image[r:r + 6, c:c + 6] += 1.0
            image[(r + 2):(r + 4), :] += 0.15
            image[:, (c + 2):(c + 4)] += 0.15
            images.append(image)
            labels.append(label)
    x = torch.tensor(np.array(images)[:, None, :, :]).to(device)
    y = torch.tensor(np.array(labels)).long().to(device)
    return x, y


def run(epochs=80, lr=0.03):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = make_dataset(device)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
    assert accuracy >= 0.95
