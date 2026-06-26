import numpy as np
from . import backend
from .log import log_function_call
from .settings import Config, log_settings, runtime_settings, using_config
from . import functional as F
from .functional.get_item import get_item
#import networkx as nx
#import matplotlib.pyplot as plt


class Tensor:

    # 确保优先级高于np.ndarray的运算符
    __array_priority__ = 200

    def __init__(
        self, data, name=None, log_enabled=None, device=None, dtype=None,
        requires_grad=None,
    ):
        if isinstance(data, Tensor):
            if requires_grad is None:
                requires_grad = data.requires_grad
            data = data.data
        self.data = backend.ensure_array(data, device=device, dtype=dtype)
        if requires_grad is None:
            requires_grad = Config.enable_backprop
        self.requires_grad = requires_grad
        self.name = name
        if log_enabled is None:
            self.log_enabled = log_settings.get('tensor_log_enabled', False)
        else:
            self.log_enabled = log_enabled
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        # Tensor-output代始终比产生它的function大一代
        self.creator = func
        self.generation = self.creator.generation + 1

    def zero_grad(self):
        self.grad = None

    def requires_grad_(self, requires_grad=True):
        self.requires_grad = requires_grad
        return self

    @log_function_call(enabled=True)
    def backward(self, retain_grad=False, create_graph=False):
        if not self.requires_grad:
            return

        if self.creator is None:
            # incaseof: x = Tensor(...); x.backward()
            return

        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            xp = backend.get_array_module(self.data)
            self.grad = Tensor(xp.ones_like(self.data), requires_grad=create_graph)

        funcs = []
        set_funcs = set()
        def add_func(f):
            if f not in set_funcs:
                funcs.append(f)
                set_funcs.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop() # pop the item with maximal generation
            if runtime_settings.get('remove_recursive_ref', True):
                gys = [output().grad for output in f.outputs] # weakref used
            else:
                gys = [output.grad for output in f.outputs]

            # create_graph默认False: 默认单次backward后不需要再次反向传播了
            # enable_backprop 为了: `inputs` 仅仅在反向传播时才需要，不反向传播时，不用保留
            # 如果create_graph 为真，表示还需要导数值，所以
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs, )

                for x, gx in zip(f.inputs, gxs):
                    if gx is None:
                        continue
                    if not x.requires_grad:
                        continue
                    if x.grad is None:
                        # in case of: y = x + x
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)

            if not retain_grad:
                # 默认不保留中间导数
                if runtime_settings.get('remove_recursive_ref', True):
                    for output in f.outputs:
                        output().grad = None
                else:
                    for output in f.outputs:
                        output.grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple,list)):
            shape = shape[0]
        return F.reshape(self, shape)

    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple,list)):
            shape = shape[0]
        return F.view(self, shape)

    def unsqueeze(self, dim):
        return F.unsqueeze(self, dim)

    def squeeze(self, dim=None, axis=None):
        return F.squeeze(self, dim=dim, axis=axis)

    def flatten(self, start_dim=0, end_dim=-1):
        return F.flatten(self, start_dim=start_dim, end_dim=end_dim)

    def repeat(self, *repeats):
        return F.repeat(self, *repeats)

    def split(self, split_size_or_sections, dim=0):
        return F.split(self, split_size_or_sections, dim=dim)

    def chunk(self, chunks, dim=0):
        return F.chunk(self, chunks, dim=dim)

    def transpose(self, dim0=None, dim1=None):
        return F.transpose(self, dim0, dim1)

    def permute(self, *dims):
        if len(dims) == 1 and isinstance(dims[0], (tuple,list)):
            dims = tuple(dims[0])
        return F.permute(self, dims)

    def sum(self, dim=None, axis=None, keepdims=False):
        axis = dim if dim is not None else axis
        return F.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, dim=None, axis=None, keepdims=False):
        axis = dim if dim is not None else axis
        return F.mean(self, axis=axis, keepdims=keepdims)

    def exp(self):
        return F.exp(self)

    def log(self):
        return F.log(self)

    def abs(self):
        return F.abs(self)

    def sqrt(self):
        return F.sqrt(self)

    def clamp(self, min=None, max=None):
        return F.clamp(self, min=min, max=max)

    def clip(self, min=None, max=None):
        return F.clip(self, min=min, max=max)

    def relu(self):
        return F.relu(self)

    def tanh(self):
        return F.tanh(self)

    def sigmoid(self):
        return F.sigmoid(self)

    def softmax(self, dim=None, axis=-1):
        return F.softmax(self, axis=axis, dim=dim)

    def log_softmax(self, dim=None, axis=-1):
        return F.log_softmax(self, axis=axis, dim=dim)

    def argmax(self, dim=None, keepdim=False, axis=None, keepdims=None):
        return F.argmax(self, dim=dim, keepdim=keepdim, axis=axis,
                        keepdims=keepdims)

    def topk(self, k, dim=None, largest=True, sorted=True):
        return F.topk(self, k, dim=dim, largest=largest, sorted=sorted)

    def masked_fill(self, mask, value):
        return F.masked_fill(self, mask, value)

    def nonzero(self, as_tuple=False):
        return F.nonzero(self, as_tuple=as_tuple)

    def index_select(self, dim, index):
        return F.index_select(self, dim, index)

    def gather(self, dim, index):
        return F.gather(self, dim, index)

    def scatter(self, dim, index, src):
        return F.scatter(self, dim, index, src)

    def scatter_add(self, dim, index, src):
        return F.scatter_add(self, dim, index, src)

    def sort(self, dim=-1, descending=False, stable=False):
        return F.sort(self, dim=dim, descending=descending, stable=stable)

    def argsort(self, dim=-1, descending=False, stable=False):
        return F.argsort(self, dim=dim, descending=descending, stable=stable)

    def max(self, dim=None, keepdim=False, axis=None, keepdims=None):
        return F.max(self, dim=dim, keepdim=keepdim, axis=axis,
                     keepdims=keepdims)

    def amax(self, dim=None, keepdim=False, axis=None, keepdims=None):
        return F.amax(self, dim=dim, keepdim=keepdim, axis=axis,
                      keepdims=keepdims)

    def amin(self, dim=None, keepdim=False, axis=None, keepdims=None):
        return F.amin(self, dim=dim, keepdim=keepdim, axis=axis,
                      keepdims=keepdims)

    def aminmax(self, dim=None, keepdim=False, axis=None, keepdims=None):
        return F.aminmax(self, dim=dim, keepdim=keepdim, axis=axis,
                         keepdims=keepdims)

    def renamed(self, name):
        self.name = name
        return self

    def numpy(self):
        return backend.as_numpy(self.data)

    def to(self, device):
        out = Tensor(backend.to_device(self.data, device), name=self.name,
                     log_enabled=self.log_enabled,
                     requires_grad=self.requires_grad)
        if self.grad is not None:
            out.grad = self.grad.to(device)
        return out

    def detach(self):
        return Tensor(self.data, name=self.name, log_enabled=self.log_enabled,
                      requires_grad=False)

    def float(self):
        return Tensor(self.data.astype('float32'), name=self.name,
                      log_enabled=self.log_enabled,
                      requires_grad=self.requires_grad)

    def long(self):
        return Tensor(self.data.astype('int64'), name=self.name,
                      log_enabled=self.log_enabled, requires_grad=False)

    def cpu(self):
        return self.to('cpu')

    def cuda(self):
        return self.to('cuda')

    @classmethod
    def _get_shape(cls, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple,list)):
            shape = shape[0]
        return shape

    @classmethod
    def randn(self, *shape, device=None, requires_grad=None):
        shape = self._get_shape(shape)
        xp = backend.array_module_for_device(device)
        return Tensor(xp.random.randn(*shape), requires_grad=requires_grad)

    @classmethod
    def zeros(self, *shape, device=None, requires_grad=None):
        shape = self._get_shape(shape)
        xp = backend.array_module_for_device(device)
        return Tensor(xp.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def ones(self, *shape, device=None, requires_grad=None):
        shape = self._get_shape(shape)
        xp = backend.array_module_for_device(device)
        return Tensor(xp.ones(shape), requires_grad=requires_grad)

    @classmethod
    def zeros_like(self, data):
        xp = backend.get_array_module(data.data if isinstance(data, Tensor) else data)
        requires_grad = data.requires_grad if isinstance(data, Tensor) else None
        return Tensor(xp.zeros(data.shape), requires_grad=requires_grad)

    @classmethod
    def ones_like(self, data):
        xp = backend.get_array_module(data.data if isinstance(data, Tensor) else data)
        requires_grad = data.requires_grad if isinstance(data, Tensor) else None
        return Tensor(xp.ones(data.shape), requires_grad=requires_grad)

    @property
    def T(self):
        '''
        tensor.T 是 PyTorch 中对二维张量进行转置的简便方法，相当于 tensor.transpose(0, 1)。
        对于三维或更高维张量，tensor.T 不会改变张量的形状。
        '''
        return F.transpose(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def size(self, dim=None):
        # torch-like, 而不是self.data.size
        if dim is None:
            return self.data.shape
        else:
            return self.data.shape[dim]

    def item(self):
        assert len(self.data.shape) == 0
        #print(type(self.data))
        #print(self.data.value)
        data = self.data
        #print(type(data))
        #print(type(data.item()))
        #print(type(self.data), self.data)
        #print(repr(self.data.item()))
        return self.data.item()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return backend.device_of(self.data)

    def __repr__(self):
        return (
            f'Tensor({str(self.data)}, name={self.name}'
            f', shape={self.shape}, requires_grad={self.requires_grad}'
            f', grad={self.grad})'
        ).replace('\n', '\n' + ' '*8)

    def __eq__(self, other):
        return F.eq(self, other)

    def __bool__(self):
        if self.data.size != 1:
            raise ValueError(
                'bool value of Tensor with more than one value is ambiguous'
            )
        return bool(self.item())


def register_ops():
    Tensor.__neg__ = F.neg
    Tensor.__abs__ = F.abs

    Tensor.__add__ = F.add
    Tensor.__radd__ = F.add
    Tensor.__sub__ = F.sub
    Tensor.__rsub__ = F.rsub

    Tensor.__mul__ = F.mul
    Tensor.__rmul__ = F.mul
    Tensor.__truediv__ = F.div
    Tensor.__rtruediv__ = F.rdiv

    Tensor.__pow__ = F.pow
    Tensor.__matmul__ = F.matmul
    Tensor.__getitem__ = get_item

    Tensor.__eq__ = F.eq
    Tensor.__ne__ = F.ne
    Tensor.__lt__ = F.lt
    Tensor.__le__ = F.le
    Tensor.__gt__ = F.gt
    Tensor.__ge__ = F.ge


def no_grad():
    return using_config('enable_backprop', False)

def ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    return Tensor(data, requires_grad=False)

def make_tensor(data):
    return Tensor(data)

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return a.shape == b.shape and np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol, equal_nan=equal_nan)

def rand(*shape, device=None, requires_grad=False):
    xp = backend.array_module_for_device(device)
    return Tensor(xp.random.rand(*shape), requires_grad=requires_grad)

def randn(*shape, device=None, requires_grad=False):
    xp = backend.array_module_for_device(device)
    return Tensor(xp.random.randn(*shape), requires_grad=requires_grad)

def _normalize_size(size):
    if isinstance(size, int):
        return (size,)
    return tuple(size)

def arange(start, end=None, step=1, device=None, dtype=None, requires_grad=False):
    if end is None:
        start, end = 0, start
    xp = backend.array_module_for_device(device)
    return Tensor(
        xp.arange(start, end, step, dtype=dtype),
        requires_grad=requires_grad,
    )

def randint(low, high=None, size=None, device=None, dtype=None,
            requires_grad=False):
    if size is None and isinstance(high, (tuple, list)):
        size = high
        high = low
        low = 0
    elif high is None:
        high = low
        low = 0
    if size is None:
        raise ValueError('randint requires a size')
    if dtype is None:
        dtype = np.int64
    xp = backend.array_module_for_device(device)
    data = xp.random.randint(low, high, size=_normalize_size(size))
    if dtype is not None:
        data = data.astype(dtype)
    return Tensor(data, requires_grad=requires_grad)

def zeros(*shape, device=None, requires_grad=False):
    return Tensor.zeros(*shape, device=device, requires_grad=requires_grad)

def ones(*shape, device=None, requires_grad=False):
    return Tensor.ones(*shape, device=device, requires_grad=requires_grad)

def zeros_like(input, device=None, dtype=None, requires_grad=False):
    data = input.data if isinstance(input, Tensor) else backend.ensure_array(input)
    target_device = input.device if isinstance(input, Tensor) and device is None else device
    target_dtype = data.dtype if dtype is None else dtype
    xp = backend.get_array_module(data) if target_device is None else backend.array_module_for_device(target_device)
    return Tensor(
        xp.zeros(data.shape, dtype=target_dtype),
        device=target_device,
        requires_grad=requires_grad,
    )

def ones_like(input, device=None, dtype=None, requires_grad=False):
    data = input.data if isinstance(input, Tensor) else backend.ensure_array(input)
    target_device = input.device if isinstance(input, Tensor) and device is None else device
    target_dtype = data.dtype if dtype is None else dtype
    xp = backend.get_array_module(data) if target_device is None else backend.array_module_for_device(target_device)
    return Tensor(
        xp.ones(data.shape, dtype=target_dtype),
        device=target_device,
        requires_grad=requires_grad,
    )

def full(size, fill_value, device=None, dtype=None, requires_grad=False):
    xp = backend.array_module_for_device(device)
    return Tensor(
        xp.full(_normalize_size(size), fill_value, dtype=dtype),
        requires_grad=requires_grad,
    )

def full_like(input, fill_value, device=None, dtype=None, requires_grad=False):
    data = input.data if isinstance(input, Tensor) else backend.ensure_array(input)
    target_device = input.device if isinstance(input, Tensor) and device is None else device
    target_dtype = data.dtype if dtype is None else dtype
    xp = backend.get_array_module(data) if target_device is None else backend.array_module_for_device(target_device)
    return Tensor(
        xp.full(data.shape, fill_value, dtype=target_dtype),
        device=target_device,
        requires_grad=requires_grad,
    )

def tensor(data, device=None, dtype=None, requires_grad=False):
    return Tensor(data, device=device, dtype=dtype,
                  requires_grad=requires_grad)

float32 = np.float32
float64 = np.float64
long = np.int64
