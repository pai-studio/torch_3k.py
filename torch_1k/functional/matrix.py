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


# Unsqueeze: future
class _Unsqueeze(Function):
    # input: 输入的张量。
    # dim: 要插入维度的位置，范围可以是 [-input.dim()-1, input.dim()]。
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        self.x_shape = x.shape
        xp = backend.get_array_module(x)
        y = xp.expand_dims(x, axis=self.dim)
        return y


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
