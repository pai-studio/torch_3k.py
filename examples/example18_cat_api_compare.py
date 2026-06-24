"""PyTorch-compatible cat/concat API example for feature fusion."""

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
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    right = torch.tensor([[5.0], [6.0]], requires_grad=True)
    fused = torch.cat([left, right], dim=1)
    doubled = torch.concat([fused, fused], dim=0)
    loss = (doubled * doubled).sum()
    loss.backward()

    return {
        "fused": _to_numpy(fused),
        "doubled_shape": tuple(doubled.shape),
        "left_grad": _to_numpy(left.grad),
        "right_grad": _to_numpy(right.grad),
    }


if __name__ == "__main__":
    result = run()
    print(f"fused={result['fused']}")
    print(f"doubled_shape={result['doubled_shape']}")
    assert np.allclose(result["fused"], [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]])
    assert result["doubled_shape"] == (4, 3)
    assert np.allclose(result["left_grad"], [[4.0, 8.0], [12.0, 16.0]])
    assert np.allclose(result["right_grad"], [[20.0], [24.0]])
