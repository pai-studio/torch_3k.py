"""Compare einsum ellipsis behavior with PyTorch in one process."""

import sys
from pathlib import Path

import numpy as np
import torch as pytorch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_1k


def _array(shape, offset=0.0):
    size = int(np.prod(shape))
    return (np.arange(size, dtype=np.float64).reshape(shape) + offset) / 11.0


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
        (
            "...ij,jk->...ik",
            [_array((2, 3, 4), 1.0), _array((4, 5), 2.0)],
        ),
        (
            "...ij,...jk->...ik",
            [_array((1, 3, 4), 1.0), _array((2, 4, 5), 2.0)],
        ),
        (
            "...qd,...kd->...qk",
            [_array((2, 3, 4), 1.0), _array((2, 5, 4), 2.0)],
        ),
        ("...ij->...", [_array((2, 3, 4), 1.0)]),
        ("...i->i", [_array((2, 3, 4), 1.0)]),
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
