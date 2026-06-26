"""Beginner-oriented torch tutorial written as executable checks.

The same code runs with torch_1k by default. Set USE_TORCH_1K=0 to run it with
PyTorch and verify the import-replacement path.
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
    from torch_1k.utils.data import DataLoader, TensorDataset

    BACKEND_NAME = "torch_1k"
else:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    BACKEND_NAME = "torch"


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def tensor_basics(device):
    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    x = torch.from_numpy(raw).to(device=device)
    same = torch.as_tensor(x)
    eye = torch.eye(3, dtype=torch.float32, device=device)
    noise = torch.randn_like(x) * 0.0

    row_sum = (x + noise).sum(dim=1, keepdim=True)
    centered = x - x.mean(dim=0, keepdim=True)
    gram = x.matmul(x.T)
    copied = gram.clone()

    assert same is x
    assert len(x) == 2
    assert x.tolist() == raw.tolist()
    assert eye.shape == (3, 3)
    assert row_sum.shape == (2, 1)
    assert centered.shape == x.shape
    assert copied.shape == gram.shape
    return {
        "row_sum": _to_numpy(row_sum),
        "gram": _to_numpy(copied),
    }


def autograd_basics(device):
    x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    weights = torch.tensor([0.5, -1.0, 2.0], device=device)
    y = (
        (x * weights).sum()
        + x.reshape(1, 3).mm(torch.eye(3, device=device)).sum()
    ) ** 2
    y.backward()
    return _to_numpy(x.grad)


def manual_linear_regression(device, steps=80, lr=0.08):
    x = torch.linspace(-1.0, 1.0, 12, device=device).reshape(12, 1)
    y = 2.0 * x - 0.5
    weight = torch.randn(1, 1, device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)

    for _ in range(steps):
        pred = x.matmul(weight) + bias
        loss = ((pred - y) ** 2).mean()
        weight.grad = None
        bias.grad = None
        loss.backward()
        weight.data = weight.data - lr * weight.grad.data
        bias.data = bias.data - lr * bias.grad.data

    final_loss = ((x.matmul(weight) + bias - y) ** 2).mean()
    return final_loss.item(), _to_numpy(weight), _to_numpy(bias)


class TinyTutorialMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def make_classification_tensors(device):
    x_np = np.array([
        [-1.0, -1.0],
        [-0.9, -0.8],
        [-1.1, -1.2],
        [1.0, 1.0],
        [0.9, 0.8],
        [1.1, 1.2],
    ], dtype=np.float32)
    y_np = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    return (
        torch.as_tensor(x_np).to(device=device, dtype=torch.float32),
        torch.as_tensor(y_np).to(device=device).long(),
    )


def train_mlp_with_dataloader(device, epochs=120, lr=0.2):
    x, y = make_classification_tensors(device)
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=True)
    model = TinyTutorialMLP().to(device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == y).float().mean().item()
    return loss.item(), accuracy, _to_numpy(pred)


def run_all():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    basics = tensor_basics(device)
    grad = autograd_basics(device)
    regression_loss, weight, bias = manual_linear_regression(device)
    mlp_loss, accuracy, pred = train_mlp_with_dataloader(device)
    return {
        "backend": BACKEND_NAME,
        "device": str(device),
        "basics": basics,
        "grad": grad,
        "regression_loss": regression_loss,
        "weight": weight,
        "bias": bias,
        "mlp_loss": mlp_loss,
        "accuracy": accuracy,
        "pred": pred,
    }


if __name__ == "__main__":
    result = run_all()
    print(f"backend={result['backend']}")
    print(f"device={result['device']}")
    print(f"row_sum={result['basics']['row_sum']}")
    print(f"grad={result['grad']}")
    print(f"regression_loss={result['regression_loss']:.6f}")
    print(f"mlp_loss={result['mlp_loss']:.6f}")
    print(f"accuracy={result['accuracy']:.6f}")
    print(f"pred={result['pred']}")
    assert result["regression_loss"] < 0.03
    assert result["accuracy"] == 1.0
