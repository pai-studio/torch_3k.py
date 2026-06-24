"""PyTorch-compatible shape API example for model forward code."""

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


def _to_numpy(x):
    return x.detach().cpu().numpy()


def run():
    x = torch.tensor(
        np.arange(24).reshape(2, 1, 3, 4).astype("float32"),
        requires_grad=True,
    )

    features = x.squeeze(1).unsqueeze(1).flatten(start_dim=1)
    viewed = features.view(features.shape[0], -1)
    module_features = nn.Flatten(start_dim=1)(x)
    loss = viewed.sum() + module_features.sum()
    loss.backward()

    return {
        "features_shape": tuple(features.shape),
        "viewed_shape": tuple(viewed.shape),
        "module_shape": tuple(module_features.shape),
        "grad": _to_numpy(x.grad),
    }


if __name__ == "__main__":
    result = run()
    print(f"features_shape={result['features_shape']}")
    print(f"viewed_shape={result['viewed_shape']}")
    print(f"module_shape={result['module_shape']}")
    assert result["features_shape"] == (2, 12)
    assert result["viewed_shape"] == (2, 12)
    assert result["module_shape"] == (2, 12)
    assert np.allclose(result["grad"], np.ones((2, 1, 3, 4)) * 2)
