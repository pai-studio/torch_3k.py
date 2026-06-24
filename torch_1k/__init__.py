from .log import log_function_call
from .function import Function

from .settings import log_settings, runtime_settings
from .functional import *
from .tensor import (
    Tensor, no_grad, allclose, arange, full, full_like, rand, randn, randint,
    ones, ones_like, tensor, zeros, zeros_like,
    float32, float64, long, register_ops
)
from .misc import *
from . import cuda

from .version import __version__

register_ops()

# 兼容 `torch.utils.data.DataLoader` 形式的访问。
from .utils import data
