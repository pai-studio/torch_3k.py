import numpy as np
import pickle
from .tensor import Tensor
from . import functional as F
from . import backend


def manual_seed(seed):
    backend.seed(seed)

def unsqueeze(input, dim):
    if isinstance(input, Tensor):
        ndim = input.ndim
        dim = dim if dim >= 0 else ndim + 1 + dim
        shape = list(input.shape)
        shape.insert(dim, 1)
        return F.reshape(input, tuple(shape))

    data = backend.ensure_array(input)
    xp = backend.get_array_module(data)
    return Tensor(xp.expand_dims(data, axis=dim), requires_grad=False)

def linspace(start, stop, steps, device=None, requires_grad=False):
    xp = backend.array_module_for_device(device)
    data = xp.linspace(start, stop, steps)
    return Tensor(data, requires_grad=requires_grad)

def normal(mean=0, std=1.0, size=None, device=None, requires_grad=False):
    xp = backend.array_module_for_device(device)
    data = xp.random.normal(loc=mean, scale=std, size=size)
    return Tensor(data, requires_grad=requires_grad)

def mean(x):
    return F.sum(x)/np.prod(x.shape)

def argmax(input, dim=None):
    data = input.data if isinstance(input, Tensor) else backend.ensure_array(input)
    xp = backend.get_array_module(data)
    return Tensor(xp.argmax(data, axis=dim), requires_grad=False)

def save(obj, f):
    with open(f, 'wb') as file:
        pickle.dump(obj, file)

def load(f):
    with open(f, 'rb') as file:
        return pickle.load(file)
