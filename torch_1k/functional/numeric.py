import numpy as np
from collections import namedtuple

from torch_1k import backend
from ..function import Function
from .matrix import sum_to


MaxResult = namedtuple('MaxResult', ['values', 'indices'])


def _adjust_grad_shape(gx, shape):
    if gx.shape != shape:
        return sum_to(gx, shape)
    return gx


# Square
class Square(Function):

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        return 2 * x * gy

def square(x):
    return Square()(x)


# Exp
class Exp(Function):

    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        # 为了计算高阶导数, 要求grad也为Tensor类型
        # x = self.inputs[0].data
        x = self.inputs[0]
        # return np.exp(x) * gy
        return exp(x) * gy

def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.log(x)

    def backward(self, gy):
        x = self.inputs[0]
        return gy / x

def log(x):
    return Log()(x)


class ReLU(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.maximum(x, 0.0)

    def backward(self, gy):
        x = self.inputs[0]
        return gy * (x.data > 0)

def relu(x):
    return ReLU()(x)


# Neg
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

# Add
class Add(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        # 发生了隐式broadcast -> sum
        y = x1 + x2
        # self.broadcast_shape = y.shape
        return y

    def backward(self, gy):
        gx1, gx2 = gy, gy
        #print(f'***{self.x1_shape=}, {self.x2_shape=}, {self.broadcast_shape=}')
        #if self.x1_shape != self.broadcast_shape:
        #    gx1 = sum_to(gx1, self.x1_shape)
        #if self.x2_shape != self.broadcast_shape:
        #    # 梯度回到自己原来的shape
        #    gx2 = sum_to(gx2, self.x2_shape)
        # 与上面等价
        if self.x1_shape != self.x2_shape:
            x1, x2 = self.inputs
            gx1 = sum_to(gx1, self.x1_shape)
            gx2 = sum_to(gx2, self.x2_shape)
        return gx1, gx2

def add(x1, x2):
    return Add()(x1, x2)


# Sub
class Sub(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        return x1 - x2

    def backward(self, gy):
        gx1 = _adjust_grad_shape(gy, self.x1_shape)
        gx2 = _adjust_grad_shape(-gy, self.x2_shape)
        return gx1, gx2

def sub(x1, x2):
    return Sub()(x1, x2)

def rsub(x1, x2):
    # swap
    return Sub()(x2, x1)


# Mul
class Mul(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        return x1 * x2

    def backward(self, gy):
        x1, x2 = self.inputs[0], self.inputs[1]
        gx1 = _adjust_grad_shape(x2*gy, self.x1_shape)
        gx2 = _adjust_grad_shape(x1*gy, self.x2_shape)
        return gx1, gx2

def mul(x1, x2):
    return Mul()(x1, x2)

# Div
class Div(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        return x1 / x2

    def backward(self, gy):
        x1, x2 = self.inputs[0], self.inputs[1]
        gx1 = _adjust_grad_shape(gy/x2, self.x1_shape)
        gx2 = _adjust_grad_shape(-gy*x1 / x2 ** 2, self.x2_shape)
        return gx1, gx2

def div(x1, x2):
    return Div()(x1, x2)

def rdiv(x1, x2):
    # swap
    return Div()(x2, x1)

# Pow
class Pow(Function):
    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x **(c-1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

# Sin
class Sin(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.sin(x)

    def backward(self, gy):
        x = self.inputs[0]
        # gx = gy * np.cos(x)
        gx = gy * cos(x) # call Cos()(x)
        return gx

def sin(x):
    return Sin()(x)

# Cos
class Cos(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.cos(x)

    def backward(self, gy):
        x = self.inputs[0]
        # gx = gy * np.sin(x)
        gx = - gy * sin(x)
        return gx

def cos(x):
    return Cos()(x)

# Tanh
class Tanh(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.tanh(x)

    def backward(self, gy):
        # 1 - y**2
        y = self.outputs[0]()
        gx = gy * (1 - y*y)
        return gx

def tanh(x):
    return Tanh()(x)

# Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return 1/(1+xp.exp(-x))

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y * (1 - y)

def sigmoid(x):
    return Sigmoid()(x)


class Maximum(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        xp = backend.get_array_module(x1, x2)
        return xp.maximum(x1, x2)

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        x1, x2 = self.inputs
        greater1 = x1.data > x2.data
        greater2 = x2.data > x1.data
        equal = x1.data == x2.data
        gx1 = Tensor(gy.data * (greater1 + equal * 0.5))
        gx2 = Tensor(gy.data * (greater2 + equal * 0.5))
        return (
            _adjust_grad_shape(gx1, self.x1_shape),
            _adjust_grad_shape(gx2, self.x2_shape),
        )

def maximum(x1, x2):
    return Maximum()(x1, x2)


class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.indices = None
        self.normalized_axis = None

    def forward(self, x):
        xp = backend.get_array_module(x)
        self.x_shape = x.shape
        if self.axis is None:
            self.normalized_axis = None
            return xp.max(x, keepdims=self.keepdims)

        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        self.normalized_axis = axis
        self.indices = xp.argmax(x, axis=axis)
        return xp.max(x, axis=axis, keepdims=self.keepdims)

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        x = self.inputs[0]
        xp = backend.get_array_module(x.data)
        if self.normalized_axis is None:
            y = self.outputs[0]().data
            mask = x.data == y
            count = xp.sum(mask)
            return Tensor(gy.data * mask / count)

        axis = self.normalized_axis
        gy_data = gy.data
        indices = self.indices
        if not self.keepdims:
            gy_data = xp.expand_dims(gy_data, axis=axis)
        indices = xp.expand_dims(indices, axis=axis)
        gx = xp.zeros_like(x.data)
        xp.put_along_axis(gx, indices, gy_data, axis=axis)
        return Tensor(gx)

def _max_indices(x, axis, keepdims=False):
    from torch_1k.tensor import Tensor

    data = x.data if isinstance(x, Tensor) else backend.ensure_array(x)
    xp = backend.get_array_module(data)
    axis = axis if axis >= 0 else data.ndim + axis
    indices = xp.argmax(data, axis=axis)
    if keepdims:
        indices = xp.expand_dims(indices, axis=axis)
    return Tensor(indices.astype('int64'), requires_grad=False)

def max(input, dim=None, keepdim=False, axis=None, keepdims=None):
    if dim is not None and not isinstance(dim, (int, np.integer)):
        return maximum(input, dim)

    axis = dim if dim is not None else axis
    if keepdims is not None:
        keepdim = keepdims
    values = Max(axis=axis, keepdims=keepdim)(input)
    if axis is None:
        return values
    return MaxResult(values=values, indices=_max_indices(input, axis, keepdim))


class Softmax(Function):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        xp = backend.get_array_module(x)
        shifted = x - xp.max(x, axis=self.axis, keepdims=True)
        exp_x = xp.exp(shifted)
        return exp_x / xp.sum(exp_x, axis=self.axis, keepdims=True)

    def backward(self, gy):
        y = self.outputs[0]()
        return y * (gy - (gy * y).sum(axis=self.axis, keepdims=True))

def softmax(x, axis=-1, dim=None):
    axis = dim if dim is not None else axis
    return Softmax(axis)(x)


class LogSoftmax(Function):
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        xp = backend.get_array_module(x)
        shifted = x - xp.max(x, axis=self.axis, keepdims=True)
        log_sum_exp = xp.log(xp.sum(xp.exp(shifted), axis=self.axis, keepdims=True))
        return shifted - log_sum_exp

    def backward(self, gy):
        y = self.outputs[0]()
        return gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)

def log_softmax(x, axis=-1, dim=None):
    axis = dim if dim is not None else axis
    return LogSoftmax(axis)(x)
