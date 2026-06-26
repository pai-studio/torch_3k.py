"""Compare amax tuple-dimension reductions with PyTorch."""

import sys
from pathlib import Path

import numpy as np
import torch as pytorch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_1k


def run():
    data = np.array([
        [
            [[1.0, 3.0], [3.0, 2.0]],
            [[0.0, 5.0], [5.0, 1.0]],
        ],
        [
            [[4.0, 4.0], [1.0, 4.0]],
            [[2.0, 6.0], [6.0, 6.0]],
        ],
    ], dtype=np.float64)

    ours = torch_1k.tensor(data, requires_grad=True)
    ours_out = torch_1k.amax(ours, dim=(2, 3), keepdim=True)
    ours_out.sum().backward()

    refs = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    refs_out = pytorch.amax(refs, dim=(2, 3), keepdim=True)
    refs_out.sum().backward()

    return {
        "values": ours_out.numpy(),
        "value_diff": np.max(np.abs(
            ours_out.numpy() - refs_out.detach().numpy(),
        )),
        "grad_diff": np.max(np.abs(
            ours.grad.numpy() - refs.grad.detach().numpy(),
        )),
    }


if __name__ == "__main__":
    result = run()
    print(f"values=\n{result['values']}")
    print(f"value_diff={result['value_diff']:.12e}")
    print(f"grad_diff={result['grad_diff']:.12e}")

    assert result["values"].shape == (2, 2, 1, 1)
    assert result["value_diff"] == 0.0
    assert result["grad_diff"] == 0.0
