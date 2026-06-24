
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


def _first_dim(tensor):
    size = getattr(tensor, 'size', None)
    if callable(size):
        return size(0)
    return tensor.shape[0]


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        if not tensors:
            raise ValueError('TensorDataset requires at least one tensor')
        first_size = _first_dim(tensors[0])
        assert all(_first_dim(tensor) == first_size for tensor in tensors), (
            'Size mismatch between tensors'
        )
        self.tensors = tuple(tensors)

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return _first_dim(self.tensors[0])
