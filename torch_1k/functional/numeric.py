import numpy as np
from collections import namedtuple

from torch_1k import backend
from ..function import Function
from .matrix import sum_to


MaxResult = namedtuple('MaxResult', ['values', 'indices'])
AminMaxResult = namedtuple('AminMaxResult', ['min', 'max'])
TopKResult = namedtuple('TopKResult', ['values', 'indices'])
SortResult = namedtuple('SortResult', ['values', 'indices'])


def _adjust_grad_shape(gx, shape):
    if gx.shape != shape:
        return sum_to(gx, shape)
    return gx


def _normalize_axis(axis, ndim, name):
    if not isinstance(axis, (int, np.integer)):
        raise TypeError(f'{name} dim must be an integer')
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise IndexError(f'{name} dim is out of range')
    return axis


def _normalize_axes(axis, ndim, name):
    if axis is None:
        return None
    if isinstance(axis, (int, np.integer)):
        axes = (int(axis),)
    elif isinstance(axis, (tuple, list)):
        axes = tuple(int(item) for item in axis)
    else:
        raise TypeError(f'{name} dim must be an integer or tuple of integers')

    normalized = []
    for item in axes:
        if item < 0:
            item += ndim
        if item < 0 or item >= ndim:
            raise IndexError(f'{name} dim is out of range')
        if item in normalized:
            raise ValueError(f'{name} dim contains duplicates')
        normalized.append(item)
    return tuple(normalized)


def _ensure_integer_index(index, name):
    if not np.issubdtype(index.dtype, np.integer):
        raise TypeError(f'{name} index must be an integer tensor')
    return index.astype('int64', copy=False)


def _array_scalar_bool(value):
    return bool(backend.as_numpy(value).item())


def _broadcast_index_selector(index_shape, axis, index, xp):
    selector = []
    ndim = len(index_shape)
    for dim, size in enumerate(index_shape):
        if dim == axis:
            selector.append(index)
            continue
        shape = [1] * ndim
        shape[dim] = size
        selector.append(xp.arange(size).reshape(tuple(shape)))
    return xp.broadcast_arrays(*selector)


def _comparison(x1, x2, op):
    from torch_1k.tensor import Tensor

    if isinstance(x1, Tensor):
        x1 = x1.data
    else:
        x1 = backend.ensure_array(x1)
    if isinstance(x2, Tensor):
        x2 = x2.data
    else:
        x2 = backend.ensure_array(x2)

    xp = backend.get_array_module(x1, x2)
    if xp is not np:
        x1 = xp.asarray(x1)
        x2 = xp.asarray(x2)
    return Tensor(op(x1, x2), requires_grad=False)


def eq(x1, x2):
    return _comparison(x1, x2, lambda a, b: a == b)


def ne(x1, x2):
    return _comparison(x1, x2, lambda a, b: a != b)


def lt(x1, x2):
    return _comparison(x1, x2, lambda a, b: a < b)


def le(x1, x2):
    return _comparison(x1, x2, lambda a, b: a <= b)


def gt(x1, x2):
    return _comparison(x1, x2, lambda a, b: a > b)


def ge(x1, x2):
    return _comparison(x1, x2, lambda a, b: a >= b)


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


class Abs(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.abs(x)

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        x = self.inputs[0]
        xp = backend.get_array_module(x.data)
        return gy * Tensor(xp.sign(x.data), requires_grad=False)

def abs(x):
    return Abs()(x)


class Sqrt(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.sqrt(x)

    def backward(self, gy):
        x = self.inputs[0]
        return gy * 0.5 / sqrt(x)

def sqrt(x):
    return Sqrt()(x)


class Clamp(Function):
    def __init__(self, min=None, max=None):
        if min is None and max is None:
            raise ValueError('clamp expects min or max')
        self.min = min
        self.max = max

    def forward(self, x):
        xp = backend.get_array_module(x)
        y = x
        if self.min is not None:
            y = xp.maximum(y, self.min)
        if self.max is not None:
            y = xp.minimum(y, self.max)
        return y

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        x = self.inputs[0]
        xp = backend.get_array_module(x.data)
        mask = xp.ones_like(x.data, dtype=bool)
        if self.min is not None:
            mask = mask & (x.data >= self.min)
        if self.max is not None:
            mask = mask & (x.data <= self.max)
        return gy * Tensor(mask, requires_grad=False)

def clamp(input, min=None, max=None):
    return Clamp(min=min, max=max)(input)


def clip(input, min=None, max=None):
    return clamp(input, min=min, max=max)


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


class Amax(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.normalized_axes = None
        self.x_shape = None

    def forward(self, x):
        xp = backend.get_array_module(x)
        self.x_shape = x.shape
        axes = _normalize_axes(self.axis, x.ndim, 'amax')
        self.normalized_axes = axes
        if axes is None:
            return xp.max(x, keepdims=self.keepdims)
        return xp.max(x, axis=axes, keepdims=self.keepdims)

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        x = self.inputs[0]
        xp = backend.get_array_module(x.data)
        axes = self.normalized_axes
        if axes is None:
            axes = tuple(range(x.ndim))

        y = self.outputs[0]().data
        gy_data = gy.data
        if not self.keepdims:
            shape = list(y.shape)
            for axis in sorted(axes):
                shape.insert(axis, 1)
            y = y.reshape(tuple(shape))
            gy_data = gy_data.reshape(tuple(shape))

        mask = x.data == y
        count = xp.sum(mask, axis=axes, keepdims=True)
        return Tensor(gy_data * mask / count)


def amax(input, dim=None, keepdim=False, axis=None, keepdims=None):
    axis = dim if dim is not None else axis
    if keepdims is not None:
        keepdim = keepdims
    return Amax(axis=axis, keepdims=keepdim)(input)


class Amin(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        self.normalized_axes = None
        self.x_shape = None

    def forward(self, x):
        xp = backend.get_array_module(x)
        self.x_shape = x.shape
        axes = _normalize_axes(self.axis, x.ndim, 'amin')
        self.normalized_axes = axes
        if axes is None:
            return xp.min(x, keepdims=self.keepdims)
        return xp.min(x, axis=axes, keepdims=self.keepdims)

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        x = self.inputs[0]
        xp = backend.get_array_module(x.data)
        axes = self.normalized_axes
        if axes is None:
            axes = tuple(range(x.ndim))

        y = self.outputs[0]().data
        gy_data = gy.data
        if not self.keepdims:
            shape = list(y.shape)
            for axis in sorted(axes):
                shape.insert(axis, 1)
            y = y.reshape(tuple(shape))
            gy_data = gy_data.reshape(tuple(shape))

        mask = x.data == y
        count = xp.sum(mask, axis=axes, keepdims=True)
        return Tensor(gy_data * mask / count)


def amin(input, dim=None, keepdim=False, axis=None, keepdims=None):
    axis = dim if dim is not None else axis
    if keepdims is not None:
        keepdim = keepdims
    return Amin(axis=axis, keepdims=keepdim)(input)


class AminMax(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        xp = backend.get_array_module(x)
        if self.axis is None:
            return (
                xp.min(x, keepdims=self.keepdims),
                xp.max(x, keepdims=self.keepdims),
            )
        axis = _normalize_axis(self.axis, x.ndim, 'aminmax')
        return (
            xp.min(x, axis=axis, keepdims=self.keepdims),
            xp.max(x, axis=axis, keepdims=self.keepdims),
        )

    def backward(self, gy_min, gy_max):
        raise RuntimeError('derivative for aminmax is not implemented')


def aminmax(input, dim=None, keepdim=False, axis=None, keepdims=None):
    axis = dim if dim is not None else axis
    if keepdims is not None:
        keepdim = keepdims
    if axis is not None and not isinstance(axis, (int, np.integer)):
        raise TypeError('aminmax dim must be an integer')
    minimum, maximum = AminMax(axis=axis, keepdims=keepdim)(input)
    return AminMaxResult(min=minimum, max=maximum)


def argmax(input, dim=None, keepdim=False, axis=None, keepdims=None):
    from torch_1k.tensor import Tensor

    axis = dim if dim is not None else axis
    if keepdims is not None:
        keepdim = keepdims
    if axis is not None and not isinstance(axis, (int, np.integer)):
        raise TypeError('argmax only supports a single integer dim')

    data = input.data if isinstance(input, Tensor) else backend.ensure_array(input)
    xp = backend.get_array_module(data)
    indices = xp.argmax(data, axis=axis)
    if axis is not None and keepdim:
        normalized_axis = axis if axis >= 0 else data.ndim + axis
        indices = xp.expand_dims(indices, axis=normalized_axis)
    return Tensor(indices.astype('int64'), requires_grad=False)


class TopK(Function):
    def __init__(self, k, axis=-1, largest=True, sorted=True):
        self.k = int(k)
        self.axis = axis
        self.largest = largest
        self.sorted = sorted
        self.normalized_axis = None
        self.indices = None
        self.x_shape = None

    def forward(self, x):
        if self.k < 0:
            raise ValueError('topk expects k >= 0')
        if x.ndim == 0:
            raise ValueError('topk expects at least 1-dimensional input')

        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        if axis < 0 or axis >= x.ndim:
            raise IndexError('topk dim is out of range')
        if self.k > x.shape[axis]:
            raise ValueError('topk k must not be larger than selected dim')

        self.normalized_axis = axis
        self.x_shape = x.shape
        xp = backend.get_array_module(x)
        order = xp.argsort(x, axis=axis)
        if self.largest:
            order = xp.flip(order, axis=axis)

        index = [slice(None)] * x.ndim
        index[axis] = slice(0, self.k)
        indices = order[tuple(index)]
        values = xp.take_along_axis(x, indices, axis=axis)

        self.indices = indices.astype('int64')
        return values

    def backward(self, gy_values):
        from torch_1k.tensor import Tensor

        xp = backend.get_array_module(gy_values.data)
        gx = xp.zeros(self.x_shape, dtype=gy_values.data.dtype)
        xp.put_along_axis(gx, self.indices, gy_values.data,
                          axis=self.normalized_axis)
        return Tensor(gx)

def topk(input, k, dim=None, largest=True, sorted=True):
    from torch_1k.tensor import Tensor

    axis = -1 if dim is None else dim
    op = TopK(k, axis=axis, largest=largest, sorted=sorted)
    values = op(input)
    indices = Tensor(op.indices, requires_grad=False)
    return TopKResult(values=values, indices=indices)


class Where(Function):
    def forward(self, condition, x, other):
        if condition.dtype != bool:
            raise TypeError('where condition must be a bool tensor')

        self.x_shape = x.shape
        self.other_shape = other.shape
        xp = backend.get_array_module(condition, x, other)
        self.condition = condition.astype(bool, copy=False)
        return xp.where(self.condition, x, other)

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        condition = self.condition
        gx = Tensor(gy.data * condition, requires_grad=False)
        gother = Tensor(gy.data * (~condition), requires_grad=False)
        return (
            None,
            _adjust_grad_shape(gx, self.x_shape),
            _adjust_grad_shape(gother, self.other_shape),
        )

def where(condition, input=None, other=None):
    from torch_1k.tensor import ensure_tensor

    if input is None and other is None:
        condition = ensure_tensor(condition)
        if condition.dtype != bool:
            raise TypeError('where condition must be a bool tensor')
        return nonzero(condition, as_tuple=True)
    if input is None or other is None:
        raise TypeError('where expects either one or three arguments')
    return Where()(condition, input, other)


def nonzero(input, as_tuple=False):
    from torch_1k.tensor import Tensor, ensure_tensor

    x = ensure_tensor(input)
    xp = backend.get_array_module(x.data)

    if x.ndim == 0:
        active = _array_scalar_bool(x.data != 0)
        size = 1 if active else 0
        if as_tuple:
            return (Tensor(xp.arange(size, dtype='int64'),
                           requires_grad=False),)
        return Tensor(xp.empty((size, 0), dtype='int64'), requires_grad=False)

    indices = tuple(index.astype('int64', copy=False)
                    for index in xp.nonzero(x.data))
    if as_tuple:
        return tuple(Tensor(index, requires_grad=False) for index in indices)
    if indices:
        stacked = xp.stack(indices, axis=1)
    else:
        stacked = xp.empty((0, 0), dtype='int64')
    return Tensor(stacked.astype('int64', copy=False), requires_grad=False)


def masked_fill(input, mask, value):
    from torch_1k.tensor import Tensor

    return where(mask, Tensor(value, requires_grad=False), input)


class IndexSelect(Function):
    def __init__(self, axis):
        self.axis = axis
        self.normalized_axis = None
        self.index = None
        self.x_shape = None

    def forward(self, x, index):
        if x.ndim == 0:
            raise ValueError('index_select expects at least 1-dimensional input')
        if index.ndim != 1:
            raise ValueError('index_select index must be 1-dimensional')

        axis = _normalize_axis(self.axis, x.ndim, 'index_select')
        index = _ensure_integer_index(index, 'index_select')

        self.normalized_axis = axis
        self.index = index
        self.x_shape = x.shape
        xp = backend.get_array_module(x, index)
        return xp.take(x, index, axis=axis)

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        xp = backend.get_array_module(gy.data)
        gx = xp.zeros(self.x_shape, dtype=gy.data.dtype)
        selector = _broadcast_index_selector(
            gy.shape, self.normalized_axis, self.index, xp
        )
        xp.add.at(gx, tuple(selector), gy.data)
        return Tensor(gx), None

def index_select(input, dim, index):
    return IndexSelect(dim)(input, index)


class Gather(Function):
    def __init__(self, axis):
        self.axis = axis
        self.normalized_axis = None
        self.index = None
        self.x_shape = None

    def forward(self, x, index):
        if x.ndim == 0:
            raise ValueError('gather expects at least 1-dimensional input')
        if index.ndim != x.ndim:
            raise ValueError('gather index must have the same rank as input')

        axis = _normalize_axis(self.axis, x.ndim, 'gather')
        index = _ensure_integer_index(index, 'gather')
        for dim, size in enumerate(index.shape):
            if dim != axis and size > x.shape[dim]:
                raise ValueError('gather index shape is out of bounds')

        self.normalized_axis = axis
        self.index = index
        self.x_shape = x.shape
        xp = backend.get_array_module(x, index)
        return xp.take_along_axis(x, index, axis=axis)

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        xp = backend.get_array_module(gy.data)
        gx = xp.zeros(self.x_shape, dtype=gy.data.dtype)
        selector = _broadcast_index_selector(
            self.index.shape, self.normalized_axis, self.index, xp
        )
        xp.add.at(gx, tuple(selector), gy.data)
        return Tensor(gx), None

def gather(input, dim, index):
    return Gather(dim)(input, index)


def _validate_scatter_inputs(x, index, src, axis, name):
    if x.ndim == 0:
        raise ValueError(f'{name} expects at least 1-dimensional input')
    if index.ndim != x.ndim:
        raise ValueError(f'{name} index must have the same rank as input')
    if src.shape != index.shape:
        raise ValueError(f'{name} src shape must match index shape')

    axis = _normalize_axis(axis, x.ndim, name)
    index = _ensure_integer_index(index, name)
    for dim, size in enumerate(index.shape):
        if dim != axis and size > x.shape[dim]:
            raise ValueError(f'{name} index shape is out of bounds')

    xp = backend.get_array_module(x, index, src)
    invalid = (index < 0) | (index >= x.shape[axis])
    if _array_scalar_bool(xp.any(invalid)):
        raise IndexError(f'{name} index is out of bounds for dim')
    return axis, index, xp


class Scatter(Function):
    def __init__(self, axis):
        self.axis = axis
        self.normalized_axis = None
        self.index = None
        self.x_shape = None
        self.src_shape = None
        self.selector = None

    def forward(self, x, index, src):
        axis, index, xp = _validate_scatter_inputs(
            x, index, src, self.axis, 'scatter'
        )
        self.normalized_axis = axis
        self.index = index
        self.x_shape = x.shape
        self.src_shape = src.shape
        self.selector = _broadcast_index_selector(index.shape, axis, index, xp)

        y = x.copy()
        y[tuple(self.selector)] = src
        return y

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        xp = backend.get_array_module(gy.data, self.index)
        gx = gy.data.copy()
        gx[tuple(self.selector)] = 0
        gsrc = xp.take_along_axis(
            gy.data, self.index, axis=self.normalized_axis
        )
        return Tensor(gx), None, Tensor(gsrc)


class ScatterAdd(Function):
    def __init__(self, axis):
        self.axis = axis
        self.normalized_axis = None
        self.index = None
        self.x_shape = None
        self.src_shape = None
        self.selector = None

    def forward(self, x, index, src):
        axis, index, xp = _validate_scatter_inputs(
            x, index, src, self.axis, 'scatter_add'
        )
        self.normalized_axis = axis
        self.index = index
        self.x_shape = x.shape
        self.src_shape = src.shape
        self.selector = _broadcast_index_selector(index.shape, axis, index, xp)

        y = x.copy()
        xp.add.at(y, tuple(self.selector), src)
        return y

    def backward(self, gy):
        from torch_1k.tensor import Tensor

        xp = backend.get_array_module(gy.data, self.index)
        gsrc = xp.take_along_axis(
            gy.data, self.index, axis=self.normalized_axis
        )
        return Tensor(gy.data), None, Tensor(gsrc)


def scatter(input, dim, index, src):
    return Scatter(dim)(input, index, src)


def scatter_add(input, dim, index, src):
    return ScatterAdd(dim)(input, index, src)


class Sort(Function):
    def __init__(self, axis=-1, descending=False, stable=False):
        self.axis = axis
        self.descending = descending
        self.stable = stable
        self.normalized_axis = None
        self.indices = None
        self.x_shape = None

    def forward(self, x):
        if not isinstance(self.axis, (int, np.integer)):
            raise TypeError('sort dim must be an integer')

        xp = backend.get_array_module(x)
        self.x_shape = x.shape

        if x.ndim == 0:
            axis = 0 if self.axis in (-1, 0) else self.axis
            if axis != 0:
                raise IndexError('sort dim is out of range')
            self.normalized_axis = None
            self.indices = xp.array(0, dtype='int64')
            return x.copy()

        axis = self.axis if self.axis >= 0 else x.ndim + self.axis
        if axis < 0 or axis >= x.ndim:
            raise IndexError('sort dim is out of range')

        self.normalized_axis = axis
        order = xp.argsort(x, axis=axis)
        if self.descending:
            order = xp.flip(order, axis=axis)
        values = xp.take_along_axis(x, order, axis=axis)

        self.indices = order.astype('int64')
        return values

    def backward(self, gy_values):
        from torch_1k.tensor import Tensor

        if self.normalized_axis is None:
            return gy_values

        xp = backend.get_array_module(gy_values.data)
        gx = xp.zeros(self.x_shape, dtype=gy_values.data.dtype)
        xp.put_along_axis(gx, self.indices, gy_values.data,
                          axis=self.normalized_axis)
        return Tensor(gx)

def sort(input, dim=-1, descending=False, stable=False):
    from torch_1k.tensor import Tensor

    op = Sort(axis=dim, descending=descending, stable=stable)
    values = op(input)
    indices = Tensor(op.indices, requires_grad=False)
    return SortResult(values=values, indices=indices)


def argsort(input, dim=-1, descending=False, stable=False):
    from torch_1k.tensor import Tensor

    data = input.data if isinstance(input, Tensor) else backend.ensure_array(input)
    if not isinstance(dim, (int, np.integer)):
        raise TypeError('argsort dim must be an integer')

    xp = backend.get_array_module(data)
    if data.ndim == 0:
        axis = 0 if dim in (-1, 0) else dim
        if axis != 0:
            raise IndexError('argsort dim is out of range')
        return Tensor(xp.array(0, dtype='int64'), requires_grad=False)

    axis = dim if dim >= 0 else data.ndim + dim
    if axis < 0 or axis >= data.ndim:
        raise IndexError('argsort dim is out of range')

    indices = xp.argsort(data, axis=axis)
    if descending:
        indices = xp.flip(indices, axis=axis)
    return Tensor(indices.astype('int64'), requires_grad=False)


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
