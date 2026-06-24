"""PyTorch-compatible argmax API example for classification inference."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
else:
    import torch


def _to_numpy(x):
    return x.detach().cpu().numpy()


def run():
    logits = torch.tensor(
        [[0.1, 2.0, 1.5], [3.0, 0.2, -1.0]],
        requires_grad=True,
    )

    pred = logits.argmax(dim=1)
    pred_keepdim = torch.argmax(logits, dim=1, keepdim=True)
    global_index = torch.argmax(logits)

    return {
        "pred": _to_numpy(pred),
        "pred_keepdim": _to_numpy(pred_keepdim),
        "global_index": int(global_index.item()),
        "requires_grad": pred.requires_grad,
    }


if __name__ == "__main__":
    result = run()
    print(f"pred={result['pred']}")
    print(f"pred_keepdim={result['pred_keepdim']}")
    print(f"global_index={result['global_index']}")
    assert np.allclose(result["pred"], [1, 0])
    assert np.allclose(result["pred_keepdim"], [[1], [0]])
    assert result["global_index"] == 3
    assert result["requires_grad"] is False
