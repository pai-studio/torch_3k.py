import numpy as np
from torch_1k.function import Function
from torch_1k.tensor import Tensor
from torch_1k import backend
from .module import Module
from .parameter import Parameter


class LayerNormFunction(Function):
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

    def forward(self, x, weight, bias):
        xp = backend.get_array_module(x)
        axes = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        self.axes = axes
        self.x_shape = x.shape
        self.mean = xp.mean(x, axis=axes, keepdims=True)
        self.var = xp.var(x, axis=axes, keepdims=True)
        self.std = xp.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) / self.std
        return self.x_hat * weight + bias

    def backward(self, gy):
        xp = backend.get_array_module(gy.data)
        _, weight, _ = self.inputs
        gy_data = gy.data
        reduce_axes = tuple(range(gy_data.ndim - len(self.normalized_shape)))
        if reduce_axes:
            gweight = xp.sum(gy_data * self.x_hat, axis=reduce_axes)
            gbias = xp.sum(gy_data, axis=reduce_axes)
        else:
            gweight = gy_data * self.x_hat
            gbias = gy_data

        dxhat = gy_data * weight.data
        count = int(np.prod(self.normalized_shape))
        sum_dxhat = xp.sum(dxhat, axis=self.axes, keepdims=True)
        sum_dxhat_xhat = xp.sum(dxhat * self.x_hat, axis=self.axes, keepdims=True)
        gx = (dxhat * count - sum_dxhat - self.x_hat * sum_dxhat_xhat) / (count * self.std)
        return Tensor(gx), Tensor(gweight), Tensor(gbias)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(np.ones(self.normalized_shape), name='W')
        self.bias = Parameter(np.zeros(self.normalized_shape), name='b')

    def forward(self, x):
        return LayerNormFunction(self.normalized_shape, self.eps)(
            x, self.weight, self.bias
        )
