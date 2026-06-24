import numpy as np
from .tensor import Tensor
from . import functional as F
from . import backend


def manual_seed(seed):
    backend.seed(seed)

def unsqueeze(input, dim):
    data = input.data if isinstance(input, Tensor) else backend.ensure_array(input)
    xp = backend.get_array_module(data)
    return Tensor(xp.expand_dims(data, axis=dim))

def linspace(start, stop, steps, device=None):
    xp = backend.array_module_for_device(device)
    data = xp.linspace(start, stop, steps)
    return Tensor(data)

def normal(mean=0, std=1.0, size=None, device=None):
    xp = backend.array_module_for_device(device)
    data = xp.random.normal(loc=mean, scale=std, size=size)
    return Tensor(data)

def mean(x):
    return F.sum(x)/np.prod(x.shape)
