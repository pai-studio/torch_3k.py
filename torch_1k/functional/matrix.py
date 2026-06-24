from torch_1k import backend
from torch_1k.function import Function
from torch_1k.tensor import Tensor, ensure_tensor
from torch_1k.utils import np_sum_to


# Reshape
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape) # 反向传播，恢复原始输入的shape

def reshape(x, shape):
    if x.shape == shape:
        return ensure_tensor(x)
    return Reshape(shape)(x)

def view(x, shape):
    return reshape(x, shape)


def _normalize_dim(dim, ndim, allow_insert=False):
    upper = ndim if allow_insert else ndim - 1
    lower = -ndim - 1 if allow_insert else -ndim
    if dim < lower or dim > upper:
        raise IndexError(
            f'Dimension out of range: expected to be in range of '
            f'[{lower}, {upper}], but got {dim}'
        )
    if dim < 0:
        dim += ndim + 1 if allow_insert else ndim
    return dim


def unsqueeze(x, dim):
    x = ensure_tensor(x)
    dim = _normalize_dim(dim, x.ndim, allow_insert=True)
    shape = list(x.shape)
    shape.insert(dim, 1)
    return reshape(x, tuple(shape))


def squeeze(x, dim=None, axis=None):
    x = ensure_tensor(x)
    dim = axis if dim is None else dim
    if dim is None:
        shape = tuple(size for size in x.shape if size != 1)
        return reshape(x, shape)

    if isinstance(dim, (tuple, list)):
        dims = [_normalize_dim(d, x.ndim) for d in dim]
        dim_set = set(dims)
        shape = [
            size for i, size in enumerate(x.shape)
            if not (i in dim_set and size == 1)
        ]
        return reshape(x, tuple(shape))

    dim = _normalize_dim(dim, x.ndim)
    if x.shape[dim] != 1:
        return ensure_tensor(x)
    shape = list(x.shape)
    shape.pop(dim)
    return reshape(x, tuple(shape))


def flatten(x, start_dim=0, end_dim=-1):
    x = ensure_tensor(x)
    ndim = x.ndim
    if ndim == 0:
        return reshape(x, (1,))

    start = _normalize_dim(start_dim, ndim)
    end = _normalize_dim(end_dim, ndim)
    if start > end:
        raise ValueError('flatten expects start_dim <= end_dim')

    flat = 1
    for size in x.shape[start:end + 1]:
        flat *= size
    shape = tuple(x.shape[:start] + (flat,) + x.shape[end + 1:])
    return reshape(x, shape)


# Transpose
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        xp = backend.get_array_module(x)
        y = xp.transpose(x, self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        inv_axes = [0] * len(self.axes)
        for i, axis in enumerate(self.axes):
            inv_axes[axis] = i
        return permute(gy, tuple(inv_axes))

def transpose(x, dim0=None, dim1=None):
    if dim0 is None and dim1 is None:
        return Transpose()(x)
    ndim = x.ndim
    dim0 = dim0 if dim0 >= 0 else ndim + dim0
    dim1 = dim1 if dim1 >= 0 else ndim + dim1
    axes = list(range(ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return Transpose(tuple(axes))(x)

def permute(x, axes):
    return Transpose(tuple(axes))(x)


class Stack(Function):
    def __init__(self, axis=0):
        self.axis = axis
        self.normalized_axis = None
        self.input_count = None

    def forward(self, *xs):
        if not xs:
            raise ValueError('stack expects a non-empty Tensor sequence')
        first_shape = xs[0].shape
        for x in xs:
            if x.shape != first_shape:
                raise ValueError('stack expects each tensor to be equal size')

        rank = len(first_shape)
        axis = self.axis if self.axis >= 0 else rank + 1 + self.axis
        if axis < 0 or axis > rank:
            raise IndexError(
                f'Dimension out of range: expected to be in range of '
                f'[-{rank + 1}, {rank}], but got {self.axis}'
            )

        self.normalized_axis = axis
        self.input_count = len(xs)
        xp = backend.get_array_module(*xs)
        return xp.stack(xs, axis=axis)

    def backward(self, gy):
        from .get_item import get_item

        outputs = []
        for i in range(self.input_count):
            index = [slice(None)] * gy.ndim
            index[self.normalized_axis] = i
            outputs.append(get_item(gy, tuple(index)))
        return tuple(outputs)

def stack(tensors, dim=0):
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError('stack expects a non-empty Tensor sequence')
    return Stack(dim)(*tensors)


class Cat(Function):
    def __init__(self, axis=0):
        self.axis = axis
        self.normalized_axis = None
        self.sizes = None
        self.input_count = None

    def forward(self, *xs):
        if not xs:
            raise ValueError('cat expects a non-empty Tensor sequence')
        ndim = xs[0].ndim
        if ndim == 0:
            raise ValueError('cat expects at least 1-dimensional tensors')
        axis = _normalize_dim(self.axis, ndim)
        base_shape = xs[0].shape
        sizes = []
        for x in xs:
            if x.ndim != ndim:
                raise ValueError('cat expects tensors to have the same number of dimensions')
            for dim, (actual, expected) in enumerate(zip(x.shape, base_shape)):
                if dim != axis and actual != expected:
                    raise ValueError('cat expects tensor shapes to match except in concat dim')
            sizes.append(x.shape[axis])

        self.normalized_axis = axis
        self.sizes = sizes
        self.input_count = len(xs)
        xp = backend.get_array_module(*xs)
        return xp.concatenate(xs, axis=axis)

    def backward(self, gy):
        from .get_item import get_item

        outputs = []
        start = 0
        for size in self.sizes:
            index = [slice(None)] * gy.ndim
            index[self.normalized_axis] = slice(start, start + size)
            outputs.append(get_item(gy, tuple(index)))
            start += size
        return tuple(outputs)

def cat(tensors, dim=0, axis=None):
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError('cat expects a non-empty Tensor sequence')
    axis = dim if axis is None else axis
    return Cat(axis)(*tensors)

def concat(tensors, dim=0, axis=None):
    return cat(tensors, dim=dim, axis=axis)

def concatenate(tensors, dim=0, axis=None):
    return cat(tensors, dim=dim, axis=axis)


# MatMul 
class MatMul(Function):

    def forward(self, x, W):
        xp = backend.get_array_module(x, W)
        y = xp.matmul(x, W)
        self.x_shape = x.shape
        self.W_shape = W.shape
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.transpose(-1, -2))
        if len(self.W_shape) == 2 and len(self.x_shape) > 2:
            x2 = reshape(x, (-1, self.x_shape[-1]))
            gy2 = reshape(gy, (-1, gy.shape[-1]))
            gW = matmul(x2.T, gy2)
        else:
            gW = matmul(x.transpose(-1, -2), gy)
        return gx, gW

def matmul(x, W):
    return MatMul()(x, W)

#Broadcast
class Broadcast(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        # e.g. (3, 2) <- (3, 1)
        self.x_shape = x.shape
        xp = backend.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        # 沿着扩展后的x_shape再加回来
        # XXX NOTE 怀疑有问题，这对sum(y) 这样椒可以的，但如果是get_item呢？y[1] 
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return ensure_tensor(x)
    return Broadcast(shape)(x)


#SumTo
class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np_sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum_to(x, shape):
    # 如果shape不同，则通过sum来缩减dim
    # 如果相同，则不必缩减
    if x.shape == shape:
        return ensure_tensor(x)
    return SumTo(shape)(x)

#Sum
class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        xp = backend.get_array_module(x)
        y = xp.sum(x, axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        if self.axis is None:
            return broadcast_to(gy, self.x_shape)

        axes = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axes = tuple(axis if axis >= 0 else len(self.x_shape) + axis for axis in axes)
        if not self.keepdims:
            shape = list(gy.shape)
            for axis in sorted(axes):
                shape.insert(axis, 1)
            gy = reshape(gy, tuple(shape))
        return broadcast_to(gy, self.x_shape)

def sum(x, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)

def mean(x, axis=None, keepdims=False):
    if axis is None:
        count = x.data.size
    else:
        axes = axis if isinstance(axis, tuple) else (axis,)
        count = 1
        for ax in axes:
            count *= x.shape[ax]
    return sum(x, axis=axis, keepdims=keepdims) / count

def linear(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t + b
    t.data = None # clear
    return y
