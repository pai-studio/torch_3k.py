"""PyTorch-compatible optimizer param group example."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
    import torch_1k.optim as optim
else:
    import torch
    import torch.optim as optim


def _to_numpy(x):
    return x.detach().cpu().numpy()


def run():
    fast = torch.tensor([1.0], requires_grad=True)
    slow = torch.tensor([1.0], requires_grad=True)
    optimizer = optim.SGD([
        {"params": [fast], "lr": 0.1},
        {"params": [slow], "lr": 0.01},
    ], lr=0.001)

    fast.grad = torch.tensor([1.0])
    slow.grad = torch.tensor([1.0])
    optimizer.step()

    state = optimizer.state_dict()
    return {
        "fast": _to_numpy(fast),
        "slow": _to_numpy(slow),
        "group_lrs": [group["lr"] for group in state["param_groups"]],
    }


if __name__ == "__main__":
    result = run()
    print(f"fast={result['fast']}")
    print(f"slow={result['slow']}")
    print(f"group_lrs={result['group_lrs']}")
    assert np.allclose(result["fast"], [0.9])
    assert np.allclose(result["slow"], [0.99])
    assert result["group_lrs"] == [0.1, 0.01]
