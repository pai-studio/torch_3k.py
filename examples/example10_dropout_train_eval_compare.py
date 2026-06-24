"""PyTorch-compatible Dropout train/eval example."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
    import torch_1k.nn as nn
else:
    import torch
    import torch.nn as nn


def run():
    torch.manual_seed(0)
    x = torch.ones(40, requires_grad=True)
    dropout = nn.Dropout(p=0.5)

    dropout.train()
    train_out = dropout(x)
    train_out.sum().backward()

    dropout.eval()
    eval_out = dropout(x)

    train_values = train_out.detach().cpu().numpy()
    eval_values = eval_out.detach().cpu().numpy()
    grad_values = x.grad.detach().cpu().numpy()
    return {
        "train_values": train_values,
        "eval_values": eval_values,
        "grad_values": grad_values,
        "train_has_zero": bool((train_values == 0.0).any()),
        "train_has_scaled": bool((train_values == 2.0).any()),
    }


if __name__ == "__main__":
    result = run()
    print(f"train_values={result['train_values']}")
    print(f"eval_values={result['eval_values']}")
    print(f"grad_values={result['grad_values']}")
    assert set(np.unique(result["train_values"])).issubset({0.0, 2.0})
    assert result["train_has_zero"]
    assert result["train_has_scaled"]
    assert np.allclose(result["eval_values"], np.ones(40))
    assert np.allclose(result["grad_values"], result["train_values"])
