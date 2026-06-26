"""Compare advanced CrossEntropyLoss behavior with PyTorch in one process."""

import sys
from pathlib import Path

import numpy as np
import torch as pytorch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch_1k
import torch_1k.nn as nn_1k
import torch_1k.nn.functional as F_1k


LOGITS_4D = np.array([
    [
        [[2.0, 0.0], [1.0, -1.0]],
        [[0.0, 1.0], [2.0, 0.5]],
        [[-1.0, 2.0], [0.0, 1.5]],
    ],
    [
        [[0.5, 2.0], [-0.5, 1.0]],
        [[1.5, -1.0], [0.5, 0.0]],
        [[-0.5, 0.0], [2.0, -1.0]],
    ],
], dtype=np.float64)
TARGET_3D = np.array([
    [[0, 2], [1, -100]],
    [[1, 0], [2, 1]],
], dtype=np.int64)
CLASS_WEIGHT = np.array([1.0, 2.0, 0.5], dtype=np.float64)


def run_torch_1k():
    logits = torch_1k.tensor(LOGITS_4D, requires_grad=True)
    target = torch_1k.tensor(TARGET_3D)
    weight = torch_1k.tensor(CLASS_WEIGHT)

    loss_none = F_1k.cross_entropy(
        logits,
        target,
        weight=weight,
        ignore_index=-100,
        reduction="none",
        label_smoothing=0.2,
    )
    loss_sum = F_1k.cross_entropy(
        logits,
        target,
        weight=weight,
        ignore_index=-100,
        reduction="sum",
        label_smoothing=0.2,
    )
    loss_mean = nn_1k.CrossEntropyLoss(
        weight=weight,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.2,
    )(logits, target)
    loss_mean.backward()

    return {
        "loss_none": loss_none.numpy(),
        "loss_sum": float(loss_sum.item()),
        "loss_mean": float(loss_mean.item()),
        "grad": logits.grad.numpy(),
    }


def run_pytorch():
    logits = pytorch.tensor(LOGITS_4D, dtype=pytorch.float64,
                            requires_grad=True)
    target = pytorch.tensor(TARGET_3D, dtype=pytorch.int64)
    weight = pytorch.tensor(CLASS_WEIGHT, dtype=pytorch.float64)

    loss_none = pytorch.nn.functional.cross_entropy(
        logits,
        target,
        weight=weight,
        ignore_index=-100,
        reduction="none",
        label_smoothing=0.2,
    )
    loss_sum = pytorch.nn.functional.cross_entropy(
        logits,
        target,
        weight=weight,
        ignore_index=-100,
        reduction="sum",
        label_smoothing=0.2,
    )
    loss_mean = pytorch.nn.CrossEntropyLoss(
        weight=weight,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.2,
    )(logits, target)
    loss_mean.backward()

    return {
        "loss_none": loss_none.detach().cpu().numpy(),
        "loss_sum": float(loss_sum.item()),
        "loss_mean": float(loss_mean.item()),
        "grad": logits.grad.detach().cpu().numpy(),
    }


if __name__ == "__main__":
    result_1k = run_torch_1k()
    result_pt = run_pytorch()

    loss_none_diff = np.max(np.abs(
        result_1k["loss_none"] - result_pt["loss_none"],
    ))
    grad_diff = np.max(np.abs(result_1k["grad"] - result_pt["grad"]))

    print(f"torch_1k loss_sum={result_1k['loss_sum']:.12f}")
    print(f"pytorch  loss_sum={result_pt['loss_sum']:.12f}")
    print(f"torch_1k loss_mean={result_1k['loss_mean']:.12f}")
    print(f"pytorch  loss_mean={result_pt['loss_mean']:.12f}")
    print(f"max loss_none diff={loss_none_diff:.12e}")
    print(f"max grad diff={grad_diff:.12e}")

    assert result_1k["loss_none"].shape == TARGET_3D.shape
    assert np.allclose(result_1k["loss_none"], result_pt["loss_none"])
    assert np.allclose(result_1k["loss_sum"], result_pt["loss_sum"])
    assert np.allclose(result_1k["loss_mean"], result_pt["loss_mean"])
    assert np.allclose(result_1k["grad"], result_pt["grad"])
