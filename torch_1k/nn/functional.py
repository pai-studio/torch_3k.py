from torch_1k import backend
from torch_1k.tensor import Tensor
from torch_1k.functional.numeric import ReLU, relu


def dropout(input, p=0.5, training=True, inplace=False):
    if p < 0 or p > 1:
        raise ValueError('dropout probability has to be between 0 and 1')
    if not training or p == 0:
        return input
    if p == 1:
        return input * 0.0

    xp = backend.get_array_module(input.data)
    keep_prob = 1.0 - p
    mask = (xp.random.rand(*input.shape) < keep_prob).astype(input.data.dtype)
    mask = mask / keep_prob
    return input * Tensor(mask, requires_grad=False)
