"""PyTorch-compatible functional loss and reduction example."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
    import torch_1k.nn as nn
    import torch_1k.nn.functional as F
else:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


def _to_numpy(x):
    return x.detach().cpu().numpy()


def run():
    logits = torch.tensor([
        [2.0, 0.0, -1.0],
        [0.5, 1.0, 3.0],
        [-1.0, 2.0, 0.0],
    ], requires_grad=True)
    target = torch.tensor([0, 2, 1])

    ce_none = F.cross_entropy(logits, target, reduction="none")
    ce_sum = nn.CrossEntropyLoss(reduction="sum")(logits, target)
    ce_mean = F.cross_entropy(logits, target, reduction="mean")
    ce_sum.backward()

    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    expected = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    mse_none = F.mse_loss(pred, expected, reduction="none")
    mse_mean = nn.MSELoss(reduction="mean")(pred, expected)

    return {
        "ce_none": _to_numpy(ce_none),
        "ce_sum": float(ce_sum.item()),
        "ce_mean": float(ce_mean.item()),
        "ce_grad": _to_numpy(logits.grad),
        "mse_none": _to_numpy(mse_none),
        "mse_mean": float(mse_mean.item()),
    }


if __name__ == "__main__":
    result = run()
    print(f"ce_none={result['ce_none']}")
    print(f"ce_sum={result['ce_sum']:.6f}")
    print(f"ce_mean={result['ce_mean']:.6f}")
    print(f"mse_mean={result['mse_mean']:.6f}")

    assert result["ce_none"].shape == (3,)
    assert np.allclose(result["ce_sum"], result["ce_none"].sum())
    assert np.allclose(result["ce_mean"], result["ce_none"].mean())
    assert result["ce_grad"].shape == (3, 3)
    assert np.allclose(
        result["mse_none"],
        np.array([[1.0, 1.0], [4.0, 9.0]]),
    )
    assert np.allclose(result["mse_mean"], 3.75)
