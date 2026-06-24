"""PyTorch-compatible topk API example."""

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
    logits = torch.tensor([
        [0.1, 3.2, 1.5, 0.0],
        [2.8, 0.4, 1.2, 3.1],
        [0.2, 0.5, 4.0, 1.1],
    ], requires_grad=True)
    target = torch.tensor([1, 0, 2])

    result = torch.topk(logits, 2, dim=1)
    loss = result.values.sum()
    loss.backward()

    top1 = result.indices[:, 0]
    top2 = result.indices
    top1_acc = (top1 == target).float().mean().item()
    top2_acc = np.mean([
        int(int(target.detach().cpu().numpy()[i]) in row)
        for i, row in enumerate(_to_numpy(top2))
    ])

    return {
        "values": _to_numpy(result.values),
        "indices": _to_numpy(result.indices),
        "grad": _to_numpy(logits.grad),
        "top1_acc": float(top1_acc),
        "top2_acc": float(top2_acc),
    }


if __name__ == "__main__":
    result = run()
    print(f"values={result['values']}")
    print(f"indices={result['indices']}")
    print(f"top1_acc={result['top1_acc']:.6f}")
    print(f"top2_acc={result['top2_acc']:.6f}")

    assert np.allclose(result["values"], [
        [3.2, 1.5],
        [3.1, 2.8],
        [4.0, 1.1],
    ])
    assert np.allclose(result["indices"], [
        [1, 2],
        [3, 0],
        [2, 3],
    ])
    assert np.allclose(result["top1_acc"], 2 / 3)
    assert np.allclose(result["top2_acc"], 1.0)
