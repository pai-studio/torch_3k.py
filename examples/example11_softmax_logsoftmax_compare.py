"""PyTorch-compatible Softmax and LogSoftmax API example."""

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
    logits = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, -1.0, 0.5]],
        requires_grad=True,
    )
    probs = logits.softmax(dim=1)
    func_probs = nn.functional.softmax(logits, dim=1)
    log_probs = nn.LogSoftmax(dim=1)(logits)
    func_log_probs = nn.functional.log_softmax(logits, dim=1)
    loss = -log_probs[0, 2] - log_probs[1, 0]
    loss.backward()

    return {
        "probs": probs.detach().cpu().numpy(),
        "func_probs": func_probs.detach().cpu().numpy(),
        "log_probs": log_probs.detach().cpu().numpy(),
        "func_log_probs": func_log_probs.detach().cpu().numpy(),
        "grad": logits.grad.detach().cpu().numpy(),
    }


if __name__ == "__main__":
    result = run()
    print(f"probs={result['probs']}")
    print(f"log_probs={result['log_probs']}")
    print(f"grad={result['grad']}")
    assert np.allclose(result["probs"].sum(axis=1), [1.0, 1.0])
    assert np.allclose(result["probs"], result["func_probs"])
    assert np.allclose(np.exp(result["log_probs"]), result["probs"])
    assert np.allclose(result["log_probs"], result["func_log_probs"])
    assert result["grad"].shape == (2, 3)
