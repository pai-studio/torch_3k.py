import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import numpy as np
import pytest
import torch_1k as torch
from torch_1k.utils.data import (
    BatchSampler, DataLoader, RandomSampler, SequentialSampler,
    SubsetRandomSampler, TensorDataset,
)


def _dataset(size=6):
    x = torch.tensor(np.arange(size).astype("float32")).reshape(size, 1)
    y = torch.tensor(np.arange(size)).long()
    return TensorDataset(x, y)


def _labels(loader):
    labels = []
    for _, batch_y in loader:
        labels.extend(batch_y.numpy().tolist())
    return labels


def test_sequential_sampler_batches_in_order():
    dataset = _dataset(5)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=SequentialSampler(dataset),
    )

    assert len(loader) == 3
    assert _labels(loader) == [0, 1, 2, 3, 4]


def test_random_sampler_uses_seed_and_covers_all_samples():
    dataset = _dataset(6)
    torch.manual_seed(0)
    first = _labels(DataLoader(dataset, batch_size=2, sampler=RandomSampler(dataset)))
    torch.manual_seed(0)
    second = _labels(DataLoader(dataset, batch_size=2, sampler=RandomSampler(dataset)))

    assert first == second
    assert sorted(first) == [0, 1, 2, 3, 4, 5]
    assert first != [0, 1, 2, 3, 4, 5]


def test_subset_random_sampler_only_loads_subset():
    dataset = _dataset(8)
    torch.manual_seed(1)
    loader = DataLoader(
        dataset,
        batch_size=2,
        sampler=SubsetRandomSampler([1, 3, 5]),
    )

    labels = _labels(loader)

    assert sorted(labels) == [1, 3, 5]
    assert len(loader) == 2


def test_batch_sampler_drop_last():
    dataset = _dataset(5)
    sampler = BatchSampler(SequentialSampler(dataset), batch_size=2, drop_last=True)
    loader = DataLoader(dataset, batch_sampler=sampler)

    batches = list(loader)

    assert len(loader) == 2
    assert [batch_y.numpy().tolist() for _, batch_y in batches] == [[0, 1], [2, 3]]


def test_custom_batch_sampler_order():
    dataset = _dataset(5)
    loader = DataLoader(dataset, batch_sampler=[[4, 2], [0, 3, 1]])

    labels = [batch_y.numpy().tolist() for _, batch_y in loader]

    assert len(loader) == 2
    assert labels == [[4, 2], [0, 3, 1]]


def test_sampler_argument_validation():
    dataset = _dataset(4)

    with pytest.raises(ValueError):
        DataLoader(dataset, batch_size=2, shuffle=True, sampler=SequentialSampler(dataset))
    with pytest.raises(ValueError):
        DataLoader(
            dataset,
            batch_size=2,
            batch_sampler=BatchSampler(SequentialSampler(dataset), 2),
        )


def test_torch_utils_data_sampler_exports():
    dataset = torch.utils.data.TensorDataset(torch.tensor([[1.0], [2.0], [3.0]]))
    sampler = torch.utils.data.SequentialSampler(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=sampler)

    (batch_x,) = next(iter(loader))

    assert batch_x.shape == (2, 1)
