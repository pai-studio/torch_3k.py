import numpy as np
from collections.abc import Mapping
from numbers import Number

from ...tensor import Tensor, tensor
from ...functional import stack


def default_collate(batch):
    elem = batch[0]

    if isinstance(elem, Tensor):
        return stack(batch, dim=0)
    if isinstance(elem, np.ndarray):
        return tensor(np.stack(batch, axis=0))
    if isinstance(elem, Number):
        return tensor(batch)
    if isinstance(elem, Mapping):
        return {
            key: default_collate([sample[key] for sample in batch])
            for key in elem
        }
    if isinstance(elem, tuple):
        return tuple(default_collate(samples) for samples in zip(*batch))
    if isinstance(elem, list):
        return [default_collate(samples) for samples in zip(*batch)]
    return batch

class DataLoader:
    
    def __init__(
        self, dataset, batch_size=1, shuffle=False, drop_last=False,
        collate_fn=None,
    ) -> None:
        '''
        drop_last: 如果设置为 True，在数据集大小不能整除 batch_size 时，丢弃最后一个不足一个批次的数据。
        '''
        if batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value')
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or default_collate
        
        self.data_size= len(dataset)
        if self.drop_last:
            self.max_iter = self.data_size // self.batch_size
        else:
            self.max_iter = (self.data_size + self.batch_size - 1) // self.batch_size
        
        self.reset()
    
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        self.reset()
        return self
    
    def mark_end(self):
        self.reset()
        raise StopIteration
            
    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
            
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size: (i+1)*batch_size]
        if self.drop_last and len(batch_index) != batch_size:
            self.reset()
            raise StopIteration
        
        batch_data = [self.dataset[i] for i in batch_index]
        self.iteration += 1
        return self.collate_fn(batch_data)
    
    def next(self):
        return self.__next__()

    def __len__(self):
        return self.max_iter
