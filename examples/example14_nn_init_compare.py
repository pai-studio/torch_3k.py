"""PyTorch-compatible nn.init example for model parameter initialization."""

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
    torch.manual_seed(0)
    linear = nn.Linear(3, 2)
    conv = nn.Conv2d(1, 2, kernel_size=3)

    nn.init.xavier_uniform_(linear.weight)
    nn.init.zeros_(linear.bias)
    nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
    nn.init.ones_(conv.bias)

    x = torch.ones(4, 3)
    y = linear(x)

    linear_bound = np.sqrt(6.0 / (3 + 2))
    conv_bound = np.sqrt(6.0 / (1 * 3 * 3))
    linear_weight = _to_numpy(linear.weight)
    conv_weight = _to_numpy(conv.weight)

    return {
        "linear_out_shape": y.shape,
        "linear_bias": _to_numpy(linear.bias),
        "conv_bias": _to_numpy(conv.bias),
        "linear_in_range": bool(np.all(np.abs(linear_weight) <= linear_bound)),
        "conv_in_range": bool(np.all(np.abs(conv_weight) <= conv_bound)),
    }


if __name__ == "__main__":
    result = run()
    print(f"linear_out_shape={result['linear_out_shape']}")
    print(f"linear_bias={result['linear_bias']}")
    print(f"conv_bias={result['conv_bias']}")
    assert result["linear_out_shape"] == (4, 2)
    assert np.allclose(result["linear_bias"], [0.0, 0.0])
    assert np.allclose(result["conv_bias"], [1.0, 1.0])
    assert result["linear_in_range"]
    assert result["conv_in_range"]
