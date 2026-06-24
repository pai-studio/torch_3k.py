import builtins

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


def _normalize_repeat_sizes(repeats):
    if len(repeats) == 1 and isinstance(repeats[0], (tuple, list)):
        repeats = tuple(repeats[0])
    if not repeats:
        raise ValueError('repeat expects at least one repeat value')

    repeats = tuple(int(repeat) for repeat in repeats)
    if any(repeat < 0 for repeat in repeats):
        raise ValueError('repeat expects non-negative repeat values')
    return repeats


class Repeat(Function):
    def __init__(self, repeats):
        self.repeats = _normalize_repeat_sizes(repeats)
        self.input_shape = None
        self.padded_shape = None

    def forward(self, x):
        if len(self.repeats) < x.ndim:
            raise ValueError(
                'repeat expects repeat values for every input dimension'
            )

        pad_dims = len(self.repeats) - x.ndim
        self.input_shape = x.shape
        self.padded_shape = (1,) * pad_dims + x.shape
        xp = backend.get_array_module(x)
        return xp.tile(x, self.repeats)

    def backward(self, gy):
        shape = []
        for repeat_count, size in zip(self.repeats, self.padded_shape):
            shape.extend((repeat_count, size))
        gx = reshape(gy, tuple(shape))
        axes = tuple(range(0, len(shape), 2))
        gx = sum(gx, axis=axes)
        return reshape(gx, self.input_shape)

def repeat(input, *repeats):
    return Repeat(repeats)(input)


def _slice_along_dim(x, dim, start, end):
    from .get_item import get_item

    index = [slice(None)] * x.ndim
    index[dim] = slice(start, end)
    return get_item(x, tuple(index))


def split(input, split_size_or_sections, dim=0):
    x = ensure_tensor(input)
    dim = _normalize_dim(dim, x.ndim)
    dim_size = x.shape[dim]

    if isinstance(split_size_or_sections, int):
        split_size = int(split_size_or_sections)
        if split_size <= 0:
            raise ValueError('split expects a positive split size')
        if dim_size == 0:
            return (_slice_along_dim(x, dim, 0, 0),)
        ranges = [
            (start, min(start + split_size, dim_size))
            for start in range(0, dim_size, split_size)
        ]
    else:
        sections = tuple(int(size) for size in split_size_or_sections)
        if any(size < 0 for size in sections):
            raise ValueError('split expects non-negative section sizes')
        if builtins.sum(sections) != dim_size:
            raise ValueError('split section sizes must sum to input size')
        ranges = []
        start = 0
        for size in sections:
            ranges.append((start, start + size))
            start += size

    return tuple(_slice_along_dim(x, dim, start, end)
                 for start, end in ranges)


def chunk(input, chunks, dim=0):
    chunks = int(chunks)
    if chunks <= 0:
        raise ValueError('chunk expects a positive chunk count')

    x = ensure_tensor(input)
    dim = _normalize_dim(dim, x.ndim)
    dim_size = x.shape[dim]
    if dim_size == 0:
        return tuple(_slice_along_dim(x, dim, 0, 0) for _ in range(chunks))

    split_size = (dim_size + chunks - 1) // chunks
    return split(x, split_size, dim=dim)


def _parse_einsum_equation(equation):
    if not isinstance(equation, str):
        raise TypeError('einsum equation must be a string')

    equation = ''.join(equation.split())
    if '...' in equation:
        raise NotImplementedError('einsum ellipsis is not supported')

    parts = equation.split('->')
    if len(parts) > 2:
        raise ValueError('einsum equation has too many arrows')

    input_part = parts[0]
    input_specs = tuple(input_part.split(','))
    if input_specs == ('',) and input_part != '':
        raise ValueError('einsum equation has invalid input specs')

    for spec in input_specs:
        if len(set(spec)) != len(spec):
            raise NotImplementedError(
                'einsum repeated labels in one input are not supported'
            )

    if len(parts) == 2:
        output_spec = parts[1]
        if ',' in output_spec:
            raise ValueError('einsum output spec cannot contain commas')
        if len(set(output_spec)) != len(output_spec):
            raise ValueError('einsum output labels must be unique')
        input_labels = set(''.join(input_specs))
        if any(label not in input_labels for label in output_spec):
            raise ValueError('einsum output labels must appear in inputs')
    else:
        label_counts = {}
        for spec in input_specs:
            for label in spec:
                label_counts[label] = label_counts.get(label, 0) + 1
        output_spec = ''.join(sorted(
            label for label, count in label_counts.items() if count == 1
        ))

    return input_specs, output_spec


def _restore_einsum_grad_shape(gx, spec, present_spec, target_shape, xp):
    aligned_shape = []
    present_axis = 0
    for label in spec:
        if label in present_spec:
            aligned_shape.append(gx.shape[present_axis])
            present_axis += 1
        else:
            aligned_shape.append(1)
    gx = gx.reshape(tuple(aligned_shape))

    for axis, (current_size, target_size) in enumerate(
        zip(gx.shape, target_shape)
    ):
        if current_size == target_size:
            continue
        if target_size == 1:
            gx = xp.sum(gx, axis=axis, keepdims=True)
        elif current_size != 1:
            raise ValueError(
                'einsum backward cannot restore broadcasted gradient shape'
            )

    if gx.shape != target_shape:
        gx = xp.broadcast_to(gx, target_shape)
    return gx


class Einsum(Function):
    def __init__(self, equation):
        self.input_specs, self.output_spec = _parse_einsum_equation(equation)
        self.forward_equation = f"{','.join(self.input_specs)}->{self.output_spec}"
        self.input_shapes = None

    def forward(self, *xs):
        if len(xs) != len(self.input_specs):
            raise ValueError(
                f'einsum expected {len(self.input_specs)} operands, '
                f'got {len(xs)}'
            )
        self.input_shapes = [x.shape for x in xs]
        xp = backend.get_array_module(*xs)
        return xp.einsum(self.forward_equation, *xs)

    def backward(self, gy):
        grads = []
        for i, (x, spec, shape) in enumerate(zip(
            self.inputs, self.input_specs, self.input_shapes
        )):
            if not x.requires_grad:
                grads.append(None)
                continue

            other_specs = []
            other_arrays = []
            for j, other in enumerate(self.inputs):
                if i == j:
                    continue
                other_specs.append(self.input_specs[j])
                other_arrays.append(other.data)

            available = set(self.output_spec)
            for other_spec in other_specs:
                available.update(other_spec)

            present_spec = ''.join(label for label in spec if label in available)
            source_specs = [self.output_spec] + other_specs
            source_arrays = [gy.data] + other_arrays
            xp = backend.get_array_module(*source_arrays)
            equation = f"{','.join(source_specs)}->{present_spec}"
            gx = xp.einsum(equation, *source_arrays)
            gx = _restore_einsum_grad_shape(
                gx, spec, present_spec, shape, xp
            )

            grads.append(Tensor(gx))
        return tuple(grads)

def einsum(equation, *operands):
    if not operands:
        raise ValueError('einsum expects at least one operand')
    return Einsum(equation)(*operands)


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
