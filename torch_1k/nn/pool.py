import numpy as np
from torch_1k.function import Function
from torch_1k.tensor import Tensor
from torch_1k import backend
from .module import Module


def _pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


class MaxPool2dFunction(Function):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)

    def forward(self, x):
        xp = backend.get_array_module(x)
        self.x_shape = x.shape
        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        y = xp.zeros((n, c, out_h, out_w), dtype=x.dtype)
        self.argmax = np.zeros((n, c, out_h, out_w, 2), dtype=np.int64)

        for i in range(out_h):
            hs = i * sh
            for j in range(out_w):
                ws = j * sw
                region = x[:, :, hs:hs + kh, ws:ws + kw]
                flat = region.reshape(n, c, -1)
                idx = backend.as_numpy(xp.argmax(flat, axis=2))
                y[:, :, i, j] = xp.max(flat, axis=2)
                self.argmax[:, :, i, j, 0] = idx // kw + hs
                self.argmax[:, :, i, j, 1] = idx % kw + ws
        return y

    def backward(self, gy):
        xp = backend.get_array_module(gy.data)
        gy_data = gy.data
        gx = xp.zeros(self.x_shape, dtype=gy_data.dtype)
        n, c, out_h, out_w = gy_data.shape
        for ni in range(n):
            for ci in range(c):
                for i in range(out_h):
                    for j in range(out_w):
                        hi, wi = self.argmax[ni, ci, i, j]
                        gx[ni, ci, hi, wi] += gy_data[ni, ci, i, j]
        return Tensor(gx)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return MaxPool2dFunction(self.kernel_size, self.stride)(x)
