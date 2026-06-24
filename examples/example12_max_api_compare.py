"""PyTorch-compatible max API example."""

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


def run():
    logits = torch.tensor(
        [[0.2, 1.5, -0.1], [2.0, 0.1, 1.2]],
        requires_grad=True,
    )
    max_result = logits.max(dim=1)
    values, indices = max_result
    floor = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    clipped = torch.max(logits, floor)
    loss = values.sum() + clipped.sum()
    loss.backward()

    return {
        "values": values.detach().cpu().numpy(),
        "indices": indices.detach().cpu().numpy(),
        "clipped": clipped.detach().cpu().numpy(),
        "grad": logits.grad.detach().cpu().numpy(),
    }


if __name__ == "__main__":
    result = run()
    print(f"values={result['values']}")
    print(f"indices={result['indices']}")
    print(f"clipped={result['clipped']}")
    print(f"grad={result['grad']}")
    assert np.allclose(result["values"], [1.5, 2.0])
    assert np.allclose(result["indices"], [1, 0])
    assert result["clipped"].shape == (2, 3)
    assert result["grad"].shape == (2, 3)
