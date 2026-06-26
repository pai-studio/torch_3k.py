import numpy as np
import pickle
from .tensor import Tensor
from . import functional as F
from . import backend


def manual_seed(seed):
    backend.seed(seed)

def unsqueeze(input, dim):
    return F.unsqueeze(input, dim)

def linspace(start, stop, steps, device=None, dtype=None, requires_grad=False):
    xp = backend.array_module_for_device(device)
    data = xp.linspace(start, stop, steps, dtype=dtype)
    return Tensor(data, requires_grad=requires_grad)

def normal(mean=0, std=1.0, size=None, device=None, dtype=None,
           requires_grad=False):
    xp = backend.array_module_for_device(device)
    data = xp.random.normal(loc=mean, scale=std, size=size)
    if dtype is not None:
        data = data.astype(dtype)
    return Tensor(data, requires_grad=requires_grad)

def sum(input, dim=None, keepdim=False, axis=None, keepdims=None):
    axis = dim if dim is not None else axis
    keepdim = keepdim if keepdims is None else keepdims
    return F.sum(input, axis=axis, keepdims=keepdim)

def mean(input, dim=None, keepdim=False, axis=None, keepdims=None):
    axis = dim if dim is not None else axis
    keepdim = keepdim if keepdims is None else keepdims
    return F.mean(input, axis=axis, keepdims=keepdim)

def mm(input, mat2):
    return F.matmul(input, mat2)

def argmax(input, dim=None, keepdim=False, axis=None, keepdims=None):
    return F.argmax(input, dim=dim, keepdim=keepdim, axis=axis,
                    keepdims=keepdims)

def save(obj, f):
    with open(f, 'wb') as file:
        pickle.dump(obj, file)

def load(f):
    with open(f, 'rb') as file:
        return pickle.load(file)
