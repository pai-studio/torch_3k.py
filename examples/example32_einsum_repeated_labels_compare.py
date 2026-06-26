"""Compare einsum repeated-label behavior with PyTorch."""

import sys
from pathlib import Path

import numpy as np
import torch as pytorch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_1k


def _array(shape, offset=0.0):
    size = int(np.prod(shape))
    return (np.arange(size, dtype=np.float64).reshape(shape) + offset) / 13.0


def _run_case(equation, arrays):
    ours = [torch_1k.tensor(array, requires_grad=True) for array in arrays]
    refs = [
        pytorch.tensor(array, dtype=pytorch.float64, requires_grad=True)
        for array in arrays
    ]

    ours_out = torch_1k.einsum(equation, *ours)
    refs_out = pytorch.einsum(equation, *refs)
    ours_out.sum().backward()
    refs_out.sum().backward()

    output_diff = np.max(np.abs(ours_out.numpy() - refs_out.detach().numpy()))
    grad_diffs = [
        np.max(np.abs(ours_tensor.grad.numpy() - ref_tensor.grad.detach().numpy()))
        for ours_tensor, ref_tensor in zip(ours, refs)
    ]
    return output_diff, grad_diffs, ours_out.shape


def run():
    cases = [
        ("ii->i", [_array((3, 3), 1.0)]),
        ("ii->", [_array((3, 3), 1.0)]),
        ("ijj->i", [_array((2, 3, 3), 1.0)]),
        ("ii,i->", [_array((3, 3), 1.0), _array((3,), 2.0)]),
        ("...ii->...i", [_array((2, 3, 3), 1.0)]),
        ("...ii->...", [_array((2, 3, 3), 1.0)]),
    ]

    results = []
    for equation, arrays in cases:
        output_diff, grad_diffs, shape = _run_case(equation, arrays)
        results.append((equation, output_diff, grad_diffs, shape))
    return results


if __name__ == "__main__":
    for equation, output_diff, grad_diffs, shape in run():
        max_grad_diff = max(grad_diffs)
        print(
            f"{equation}: shape={shape}, "
            f"output_diff={output_diff:.12e}, "
            f"max_grad_diff={max_grad_diff:.12e}"
        )
        assert output_diff < 1e-10
        assert max_grad_diff < 1e-10
