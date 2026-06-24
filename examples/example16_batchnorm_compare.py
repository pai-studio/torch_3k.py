"""PyTorch-compatible BatchNorm train/eval example."""

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
    x = torch.tensor(
        np.arange(48).reshape(4, 3, 2, 2).astype("float32"),
        requires_grad=True,
    )
    bn = nn.BatchNorm2d(3)

    bn.train()
    y_train = bn(x)
    loss = (y_train * y_train).mean()
    loss.backward()
    running_mean = _to_numpy(bn.running_mean)
    weight_grad = _to_numpy(bn.weight.grad)

    bn.eval()
    y_eval = bn(x)

    return {
        "train_shape": tuple(y_train.shape),
        "eval_shape": tuple(y_eval.shape),
        "train_channel_mean": _to_numpy(y_train).mean(axis=(0, 2, 3)),
        "running_mean": running_mean,
        "weight_grad": weight_grad,
        "eval_differs_from_train": not np.allclose(_to_numpy(y_train), _to_numpy(y_eval)),
    }


if __name__ == "__main__":
    result = run()
    print(f"train_shape={result['train_shape']}")
    print(f"eval_shape={result['eval_shape']}")
    print(f"running_mean={result['running_mean']}")
    assert result["train_shape"] == (4, 3, 2, 2)
    assert result["eval_shape"] == (4, 3, 2, 2)
    assert np.allclose(result["train_channel_mean"], [0.0, 0.0, 0.0], atol=1e-6)
    assert result["running_mean"].shape == (3,)
    assert result["weight_grad"].shape == (3,)
    assert result["eval_differs_from_train"]
