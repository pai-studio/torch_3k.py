"""PyTorch-compatible linear regression example.

Switch between PyTorch and torch_1k by changing only the imports below.
The training code intentionally stays identical.
"""

import sys
import os
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


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def run(epochs=300, lr=0.01):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.unsqueeze(torch.linspace(-10, 10, 100), dim=1).to(device)
    y = 3.0 * x + 2.0 + torch.normal(0, 1, size=x.size()).to(device)

    model = LinearRegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x)
        final_loss = criterion(pred, y)

    weight = model.linear.weight.detach().cpu().numpy()
    bias = model.linear.bias.detach().cpu().numpy()
    return final_loss.item(), weight, bias


if __name__ == "__main__":
    loss, weight, bias = run()
    print(f"final_loss={loss:.6f}")
    print(f"weight={weight}")
    print(f"bias={bias}")
    assert loss < 1.5
