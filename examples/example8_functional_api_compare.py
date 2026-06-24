"""PyTorch-compatible functional API example."""

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
    x = torch.tensor(
        [[-2.0, -0.5, 0.0], [0.5, 2.0, 4.0]],
        dtype=torch.float32,
    )
    positive = torch.relu(x) + 1.0
    logged = positive.log()
    activated = nn.functional.relu(logged - 0.25)
    score = activated.mean()
    return score.item(), activated.detach().cpu().numpy()


if __name__ == "__main__":
    score, values = run()
    print(f"score={score:.6f}")
    print(f"values={values}")
    assert values.shape == (2, 3)
    assert np.isfinite(values).all()
    assert score > 0.1
