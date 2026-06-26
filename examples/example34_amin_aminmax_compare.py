import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch as pytorch
import torch_1k


def main():
    data = np.array([
        [
            [[0.2, -1.0, 0.5], [-1.0, 0.7, 0.1]],
            [[2.0, 1.5, -0.4], [0.3, -0.4, 1.2]],
        ],
        [
            [[-2.0, -2.0, 0.8], [0.4, 0.9, 1.1]],
            [[1.0, -0.8, -0.8], [2.2, 0.0, 0.6]],
        ],
    ], dtype=np.float64)

    ours = torch_1k.tensor(data, requires_grad=True)
    refs = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)

    ours_min = torch_1k.amin(ours, dim=(2, 3), keepdim=True)
    refs_min = pytorch.amin(refs, dim=(2, 3), keepdim=True)
    ours_min.sum().backward()
    refs_min.sum().backward()

    assert np.allclose(ours_min.numpy(), refs_min.detach().numpy())
    assert np.allclose(ours.grad.numpy(), refs.grad.detach().numpy())

    ours_range = torch_1k.aminmax(ours.detach(), dim=1, keepdim=True)
    refs_range = pytorch.aminmax(refs.detach(), dim=1, keepdim=True)

    assert np.allclose(ours_range.min.numpy(), refs_range.min.numpy())
    assert np.allclose(ours_range.max.numpy(), refs_range.max.numpy())

    print("amin / aminmax outputs match PyTorch")


if __name__ == "__main__":
    main()
