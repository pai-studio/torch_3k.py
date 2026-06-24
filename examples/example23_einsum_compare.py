"""PyTorch-compatible einsum API example."""

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
    query = torch.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [2.0, 0.0]],
    ], requires_grad=True)
    key = torch.tensor([
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    ], requires_grad=True)
    value = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]],
    ], requires_grad=True)

    scores = torch.einsum("bqd,bkd->bqk", query, key)
    context = torch.einsum("bqk,bkd->bqd", scores, value)
    loss = context.sum()
    loss.backward()

    return {
        "scores": _to_numpy(scores),
        "context": _to_numpy(context),
        "query_grad": _to_numpy(query.grad),
        "key_grad": _to_numpy(key.grad),
        "value_grad": _to_numpy(value.grad),
        "loss": float(loss.item()),
    }


if __name__ == "__main__":
    result = run()
    print(f"scores_shape={result['scores'].shape}")
    print(f"context_shape={result['context'].shape}")
    print(f"loss={result['loss']:.6f}")

    assert result["scores"].shape == (2, 2, 3)
    assert result["context"].shape == (2, 2, 2)
    assert result["query_grad"].shape == (2, 2, 2)
    assert result["key_grad"].shape == (2, 3, 2)
    assert result["value_grad"].shape == (2, 3, 2)
    assert np.isfinite(result["loss"])
