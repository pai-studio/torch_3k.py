"""PyTorch-compatible split/chunk/repeat shape API example."""

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
    x = torch.arange(6).float().view(2, 3)
    repeated = x.repeat(2, 1)
    top, bottom = repeated.chunk(2, dim=0)
    left, right = torch.split(top, [1, 2], dim=1)
    score = left.sum() + right.sum() + bottom.sum()

    return {
        "repeated": _to_numpy(repeated),
        "top": _to_numpy(top),
        "bottom": _to_numpy(bottom),
        "left": _to_numpy(left),
        "right": _to_numpy(right),
        "score": float(score.item()),
    }


if __name__ == "__main__":
    result = run()
    print(f"repeated={result['repeated']}")
    print(f"score={result['score']:.6f}")

    assert result["repeated"].shape == (4, 3)
    assert np.allclose(result["top"], [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    assert np.allclose(result["bottom"], result["top"])
    assert np.allclose(result["left"], [[0.0], [3.0]])
    assert np.allclose(result["right"], [[1.0, 2.0], [4.0, 5.0]])
    assert np.allclose(result["score"], 30.0)
