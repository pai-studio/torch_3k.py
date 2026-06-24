"""Compare sort / argsort behavior with PyTorch."""

import sys
from pathlib import Path

import numpy as np
import torch as pytorch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_1k


def _to_numpy(x):
    return x.detach().cpu().numpy()


def _run(torch_module):
    logits = torch_module.tensor([
        [0.1, 3.2, 1.5, 0.0],
        [2.8, 0.4, 1.2, 3.1],
        [0.2, 0.5, 4.0, 1.1],
    ], requires_grad=True)
    weights = torch_module.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ])

    ascending = torch_module.sort(logits, dim=1)
    descending = torch_module.sort(logits, dim=1, descending=True)
    ranking = torch_module.argsort(logits, dim=1, descending=True)

    loss = (descending.values * weights).sum()
    loss.backward()

    return {
        "ascending_values": _to_numpy(ascending.values),
        "ascending_indices": _to_numpy(ascending.indices),
        "descending_values": _to_numpy(descending.values),
        "descending_indices": _to_numpy(descending.indices),
        "ranking": _to_numpy(ranking),
        "grad": _to_numpy(logits.grad),
    }


def _max_diff(a, b):
    return float(np.max(np.abs(a - b)))


if __name__ == "__main__":
    mini = _run(torch_1k)
    ref = _run(pytorch)

    for key in mini:
        diff = _max_diff(mini[key], ref[key])
        print(f"{key}_max_diff={diff:.3e}")
        assert np.allclose(mini[key], ref[key])

    print(f"descending_values={mini['descending_values']}")
    print(f"ranking={mini['ranking']}")
    print(f"grad={mini['grad']}")
