import numpy as np


class Sampler:
    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(np.random.permutation(len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)


class SubsetRandomSampler(Sampler):
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        order = np.random.permutation(len(self.indices))
        return iter([self.indices[i] for i in order])

    def __len__(self):
        return len(self.indices)


class BatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last=False):
        if batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        sampler_len = len(self.sampler)
        if self.drop_last:
            return sampler_len // self.batch_size
        return (sampler_len + self.batch_size - 1) // self.batch_size
