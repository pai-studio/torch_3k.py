from torch_1k import backend
from torch_1k.function import Function


class ReLU(Function):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def forward(self, x):
        xp = backend.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)
