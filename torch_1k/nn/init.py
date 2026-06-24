import math

from torch_1k import backend
from torch_1k.tensor import Tensor


def _check_tensor(tensor):
    if not isinstance(tensor, Tensor):
        raise TypeError('nn.init functions expect a Tensor')
    return tensor


def _calculate_fan_in_and_fan_out(tensor):
    tensor = _check_tensor(tensor)
    fan_in = getattr(tensor, '_fan_in', None)
    fan_out = getattr(tensor, '_fan_out', None)
    if fan_in is not None and fan_out is not None:
        return fan_in, fan_out

    shape = tensor.shape
    if len(shape) < 2:
        raise ValueError('fan in and fan out require at least 2 dimensions')

    if len(shape) == 2:
        # 默认按 PyTorch 原始二维权重布局推断；torch_1k.Linear 会写入元数据覆盖它。
        return shape[1], shape[0]

    receptive_field_size = 1
    for size in shape[2:]:
        receptive_field_size *= size
    return shape[1] * receptive_field_size, shape[0] * receptive_field_size


def calculate_gain(nonlinearity, param=None):
    if nonlinearity in ('linear', 'sigmoid', 'conv1d', 'conv2d', 'conv3d'):
        return 1.0
    if nonlinearity == 'tanh':
        return 5.0 / 3.0
    if nonlinearity == 'relu':
        return math.sqrt(2.0)
    if nonlinearity == 'leaky_relu':
        negative_slope = 0.01 if param is None else param
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    raise ValueError(f'unsupported nonlinearity: {nonlinearity}')


def constant_(tensor, val):
    tensor = _check_tensor(tensor)
    tensor.data[...] = val
    return tensor


def zeros_(tensor):
    return constant_(tensor, 0)


def ones_(tensor):
    return constant_(tensor, 1)


def uniform_(tensor, a=0.0, b=1.0):
    tensor = _check_tensor(tensor)
    xp = backend.get_array_module(tensor.data)
    tensor.data[...] = xp.random.uniform(a, b, size=tensor.shape)
    return tensor


def normal_(tensor, mean=0.0, std=1.0):
    tensor = _check_tensor(tensor)
    xp = backend.get_array_module(tensor.data)
    tensor.data[...] = xp.random.normal(mean, std, size=tensor.shape)
    return tensor


def xavier_uniform_(tensor, gain=1.0):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(tensor, -bound, bound)


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        fan = fan_in
    elif mode == 'fan_out':
        fan = fan_out
    else:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")
    if fan <= 0:
        raise ValueError('fan must be positive')

    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)
