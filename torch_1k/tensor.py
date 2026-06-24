import numpy as np
from . import utils
from . import backend
from .log import log_function_call
from .settings import log_settings, runtime_settings, using_config
from . import functional as F
from .functional.get_item import get_item
#import networkx as nx
#import matplotlib.pyplot as plt


class Tensor:

    # 确保优先级高于np.ndarray的运算符
    __array_priority__ = 200

    def __init__(self, data, name=None, log_enabled=None, device=None, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        self.data = backend.ensure_array(data, device=device, dtype=dtype)
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

    @log_function_call(enabled=True)
    def backward(self, retain_grad=False, create_graph=False):
        if self.creator is None:
            # incaseof: x = Tensor(...); x.backward()
            return

        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            self.grad = Tensor(np.ones_like(self.data))

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

    def transpose(self):
        return F.transpose(self)

    def sum(self):
        return F.sum(self)

    def renamed(self, name):
        self.name = name
        return self

    def numpy(self):
        return backend.as_numpy(self.data)

    def to(self, device):
        out = Tensor(backend.to_device(self.data, device), name=self.name,
                     log_enabled=self.log_enabled)
        if self.grad is not None:
            out.grad = self.grad.to(device)
        return out

    def detach(self):
        return Tensor(self.data, name=self.name, log_enabled=self.log_enabled)

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
    def randn(self, *shape, device=None):
        shape = self._get_shape(shape)
        xp = backend.array_module_for_device(device)
        return Tensor(xp.random.randn(*shape))

    @classmethod
    def zeros(self, *shape, device=None):
        shape = self._get_shape(shape)
        xp = backend.array_module_for_device(device)
        return Tensor(xp.zeros(shape))

    @classmethod
    def ones(self, *shape, device=None):
        shape = self._get_shape(shape)
        xp = backend.array_module_for_device(device)
        return Tensor(xp.ones(shape))

    @classmethod
    def zeros_like(self, data):
        xp = backend.get_array_module(data.data if isinstance(data, Tensor) else data)
        return Tensor(xp.zeros(data.shape))

    @classmethod
    def ones_like(self, data):
        xp = backend.get_array_module(data.data if isinstance(data, Tensor) else data)
        return Tensor(xp.ones(data.shape))

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
            f', shape={self.shape}, grad={self.grad})'
        ).replace('\n', '\n' + ' '*8)


def register_ops():
    Tensor.__neg__ = F.neg

    Tensor.__add__ = F.add
    Tensor.__radd__ = F.add
    Tensor.__sub__ = F.sub
    Tensor.__rsub__ = F.rsub

    Tensor.__mul__ = F.mul
    Tensor.__rmul__ = F.mul
    Tensor.__truediv__ = F.div
    Tensor.__rtruediv__ = F.rdiv

    Tensor.__pow__ = F.pow
    Tensor.__getitem__ = get_item


def no_grad():
    return using_config('enable_backprop', False)

def ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    return Tensor(data)

def make_tensor(data):
    return Tensor(data)

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a = ensure_tensor(a)
    b = ensure_tensor(b)
    return a.shape == b.shape and np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol, equal_nan=equal_nan)

def rand(*shape, device=None):
    xp = backend.array_module_for_device(device)
    return Tensor(xp.random.rand(*shape))

def randn(*shape, device=None):
    xp = backend.array_module_for_device(device)
    return Tensor(xp.random.randn(*shape))

def randint(*shape, device=None):
    xp = backend.array_module_for_device(device)
    return Tensor(xp.random.randint(*shape))

def zeros(*shape, device=None):
    return Tensor.zeros(*shape, device=device)

def ones(*shape, device=None):
    return Tensor.ones(*shape, device=device)

def tensor(data, device=None, dtype=None):
    return Tensor(data, device=device, dtype=dtype)
