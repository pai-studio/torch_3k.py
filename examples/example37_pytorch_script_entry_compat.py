"""PyTorch script-entry compatibility example.

The model and training loop use common script setup idioms that often appear
before the core math runs. Set USE_TORCH_1K=0 to run the same code with
PyTorch.
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

    BACKEND_NAME = "torch_1k"
else:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    BACKEND_NAME = "torch"


class EntryCompatMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(2, 8),
            nn.Identity(),
            nn.ReLU(),
            nn.Linear(8, 2),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def make_dataset(device):
    x_np = np.array([
        [-1.0, -1.0],
        [-0.9, -1.1],
        [-1.1, -0.8],
        [1.0, 1.0],
        [0.9, 1.1],
        [1.1, 0.8],
    ], dtype="float64")
    y_np = np.array([0, 0, 0, 1, 1, 1], dtype="int64")
    x = torch.tensor(x_np).to(device=device, dtype=torch.float32)
    y = torch.tensor(y_np).to(device=device).long()
    return x, y


def run(epochs=90, lr=0.2):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = make_dataset(device)

    scratch = torch.randn(2, 3, device=device, dtype=torch.float32)
    assert scratch.dim() == 2
    assert scratch.numel() == 6
    assert scratch.is_cuda == (str(device).startswith("cuda"))
    assert x.dtype == torch.float32

    model = EntryCompatMLP().to(device=device, dtype=torch.float32)
    direct_children = list(model.children())
    module_names = [name for name, _ in model.named_modules()]
    assert len(direct_children) == 1
    assert "layers" in module_names
    assert "layers.0" in module_names

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

    model.zero_grad(set_to_none=False)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert np.allclose(_to_numpy(parameter.grad), 0.0)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        accuracy = (pred == y).float().mean().item()

    copied = x.to(logits)
    assert copied.device == logits.device
    assert copied.dtype == logits.dtype

    return {
        "backend": BACKEND_NAME,
        "device": str(device),
        "loss": loss.item(),
        "accuracy": accuracy,
        "pred": _to_numpy(pred),
        "module_count": len(list(model.modules())),
        "named_modules": module_names,
    }


if __name__ == "__main__":
    result = run()
    print(f"backend={result['backend']}")
    print(f"device={result['device']}")
    print(f"loss={result['loss']:.6f}")
    print(f"accuracy={result['accuracy']:.6f}")
    print(f"pred={result['pred']}")
    print(f"module_count={result['module_count']}")
    assert result["accuracy"] == 1.0
