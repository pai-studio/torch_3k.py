"""PyTorch-compatible requires_grad example."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
    import torch_1k.nn as nn
else:
    import torch
    import torch.nn as nn


def run():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = (x * x).sum()
    y.backward()

    data = torch.tensor([[0.2, 0.8], [0.7, 0.1]])
    model = nn.Linear(2, 1)
    loss = model(data).sum()
    loss.backward()

    with torch.no_grad():
        frozen = model(data)

    grad = x.grad.detach().cpu().numpy()
    return {
        "input_grad": grad,
        "data_requires_grad": data.requires_grad,
        "data_grad_is_none": data.grad is None,
        "weight_grad_shape": tuple(model.weight.grad.shape),
        "frozen_requires_grad": frozen.requires_grad,
    }


if __name__ == "__main__":
    result = run()
    print(result)
    assert np.allclose(result["input_grad"], [2.0, 4.0, 6.0])
    assert result["data_requires_grad"] is False
    assert result["data_grad_is_none"] is True
    assert result["frozen_requires_grad"] is False
