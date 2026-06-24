"""PyTorch-compatible DataLoader sampler / batch_sampler example."""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

USE_TORCH_1K = os.getenv("USE_TORCH_1K", "1") != "0"

if USE_TORCH_1K:
    import torch_1k as torch
    from torch_1k.utils.data import (
        BatchSampler, DataLoader, SequentialSampler, TensorDataset,
    )
else:
    import torch
    from torch.utils.data import (
        BatchSampler, DataLoader, SequentialSampler, TensorDataset,
    )


def _to_numpy(x):
    return x.detach().cpu().numpy()


def run():
    x = torch.tensor(np.arange(6).reshape(6, 1).astype("float32"))
    y = torch.tensor(np.arange(6)).long()
    dataset = TensorDataset(x, y)

    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)

    batches = []
    for batch_x, batch_y in loader:
        batches.append((_to_numpy(batch_x), _to_numpy(batch_y)))
    return batches


if __name__ == "__main__":
    result = run()
    labels = [batch_y.tolist() for _, batch_y in result]
    print(f"labels={labels}")
    assert labels == [[0, 1], [2, 3], [4, 5]]
