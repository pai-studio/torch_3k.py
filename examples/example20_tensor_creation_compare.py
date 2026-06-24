"""PyTorch-compatible tensor creation API example."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
else:
    import torch


def _to_numpy(x):
    return x.detach().cpu().numpy()


def run():
    torch.manual_seed(0)
    positions = torch.arange(0, 6, 2)
    labels = torch.randint(0, 3, (4,))
    base = torch.full((2, 3), 2.0)
    mask = torch.ones_like(base)
    zeros = torch.zeros_like(base)
    filled = torch.full_like(base, 7.0)

    return {
        "positions": _to_numpy(positions),
        "labels_shape": tuple(labels.shape),
        "base": _to_numpy(base),
        "mask": _to_numpy(mask),
        "zeros": _to_numpy(zeros),
        "filled": _to_numpy(filled),
    }


if __name__ == "__main__":
    result = run()
    print(f"positions={result['positions']}")
    print(f"labels_shape={result['labels_shape']}")
    assert np.allclose(result["positions"], [0, 2, 4])
    assert result["labels_shape"] == (4,)
    assert np.allclose(result["base"], np.full((2, 3), 2.0))
    assert np.allclose(result["mask"], np.ones((2, 3)))
    assert np.allclose(result["zeros"], np.zeros((2, 3)))
    assert np.allclose(result["filled"], np.full((2, 3), 7.0))
