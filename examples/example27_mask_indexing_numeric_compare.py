"""Compare mask, gather and numeric utility APIs with PyTorch."""

import sys
from pathlib import Path

import numpy as np
import torch as pytorch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_1k


def _to_numpy(x):
    return x.detach().cpu().numpy()


def _run(torch_module):
    scores = torch_module.tensor([
        [0.2, 2.5, 1.0, -0.5],
        [3.0, -1.0, 0.4, 2.0],
        [1.2, 0.8, 4.5, -2.0],
    ], requires_grad=True)
    labels = torch_module.tensor([[1], [3], [2]])
    invalid = torch_module.tensor([
        [False, False, True, False],
        [False, True, False, False],
        [False, False, False, True],
    ])

    masked = scores.masked_fill(invalid, -100.0)
    target_scores = masked.gather(1, labels)
    positive_scores = torch_module.where(masked > 0.0, masked, torch_module.tensor(0.0))
    protected = torch_module.sqrt(torch_module.clamp(target_scores + 101.0, min=1.0))
    bounded_energy = torch_module.abs(masked.clamp(min=-2.0, max=3.0)).sum()
    loss = protected.sum() + positive_scores.sum() * 0.1 + bounded_energy * 0.01
    loss.backward()

    return {
        "masked": _to_numpy(masked),
        "target_scores": _to_numpy(target_scores),
        "positive_scores": _to_numpy(positive_scores),
        "protected": _to_numpy(protected),
        "grad": _to_numpy(scores.grad),
    }


def _max_diff(a, b):
    return float(np.max(np.abs(a - b)))


if __name__ == "__main__":
    mini = _run(torch_1k)
    ref = _run(pytorch)

    for key in mini:
        diff = _max_diff(mini[key], ref[key])
        print(f"{key}_max_diff={diff:.3e}")
        assert np.allclose(mini[key], ref[key], atol=1e-6)

    print(f"target_scores={mini['target_scores']}")
    print(f"grad={mini['grad']}")
