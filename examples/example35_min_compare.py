import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch as pytorch
import torch_1k


def main():
    data = np.array([
        [0.4, -1.2, -1.2, 2.0],
        [3.0, 0.5, -0.7, -0.1],
    ], dtype=np.float64)
    cap = np.array([0.0, -2.0, -1.2, 1.5], dtype=np.float64)

    ours = torch_1k.tensor(data, requires_grad=True)
    ours_cap = torch_1k.tensor(cap, requires_grad=True)
    ours_min = torch_1k.min(ours, dim=1)
    ours_elementwise = torch_1k.minimum(ours, ours_cap)
    (ours_min.values.sum() + ours_elementwise.sum()).backward()

    refs = pytorch.tensor(data, dtype=pytorch.float64, requires_grad=True)
    refs_cap = pytorch.tensor(cap, dtype=pytorch.float64, requires_grad=True)
    refs_min = pytorch.min(refs, dim=1)
    refs_elementwise = pytorch.minimum(refs, refs_cap)
    (refs_min.values.sum() + refs_elementwise.sum()).backward()

    assert np.allclose(ours_min.values.numpy(), refs_min.values.detach().numpy())
    assert np.allclose(ours_min.indices.numpy(), refs_min.indices.detach().numpy())
    assert np.allclose(ours_elementwise.numpy(), refs_elementwise.detach().numpy())
    assert np.allclose(ours.grad.numpy(), refs.grad.detach().numpy())
    assert np.allclose(ours_cap.grad.numpy(), refs_cap.grad.detach().numpy())

    print("min / minimum outputs and gradients match PyTorch")


if __name__ == "__main__":
    main()
