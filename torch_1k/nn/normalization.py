import numpy as np
from torch_1k.function import Function
from torch_1k.tensor import Tensor, ensure_tensor
from torch_1k import backend
from .module import Module
from .parameter import Parameter


class LayerNormFunction(Function):
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

    def forward(self, x, weight, bias):
        xp = backend.get_array_module(x)
        axes = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        self.axes = axes
        self.x_shape = x.shape
        self.mean = xp.mean(x, axis=axes, keepdims=True)
        self.var = xp.var(x, axis=axes, keepdims=True)
        self.std = xp.sqrt(self.var + self.eps)
        self.x_hat = (x - self.mean) / self.std
        return self.x_hat * weight + bias

    def backward(self, gy):
        xp = backend.get_array_module(gy.data)
        _, weight, _ = self.inputs
        gy_data = gy.data
        reduce_axes = tuple(range(gy_data.ndim - len(self.normalized_shape)))
        if reduce_axes:
            gweight = xp.sum(gy_data * self.x_hat, axis=reduce_axes)
            gbias = xp.sum(gy_data, axis=reduce_axes)
        else:
            gweight = gy_data * self.x_hat
            gbias = gy_data

        dxhat = gy_data * weight.data
        count = int(np.prod(self.normalized_shape))
        sum_dxhat = xp.sum(dxhat, axis=self.axes, keepdims=True)
        sum_dxhat_xhat = xp.sum(dxhat * self.x_hat, axis=self.axes, keepdims=True)
        gx = (dxhat * count - sum_dxhat - self.x_hat * sum_dxhat_xhat) / (count * self.std)
        return Tensor(gx), Tensor(gweight), Tensor(gbias)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(np.ones(self.normalized_shape), name='W')
        self.bias = Parameter(np.zeros(self.normalized_shape), name='b')

    def forward(self, x):
        return LayerNormFunction(self.normalized_shape, self.eps)(
            x, self.weight, self.bias
        )


class _BatchNorm(Module):
    def __init__(
        self, num_features, eps=1e-5, momentum=0.1, affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Parameter(np.ones(num_features), name='W')
            self.bias = Parameter(np.zeros(num_features), name='b')
        else:
            self.weight = None
            self.bias = None
        if track_running_stats:
            self.register_buffer('running_mean', Tensor(np.zeros(num_features),
                                                        requires_grad=False))
            self.register_buffer('running_var', Tensor(np.ones(num_features),
                                                       requires_grad=False))

    def _check_input_dim(self, x):
        raise NotImplementedError()

    def _feature_shape(self, ndim):
        shape = [1] * ndim
        shape[1] = self.num_features
        return tuple(shape)

    def _batch_axes(self, x):
        return tuple([0] + list(range(2, x.ndim)))

    def _update_running_stats(self, x, mean, var, axes):
        mean_data = mean.data.reshape(self.num_features)
        var_data = var.data.reshape(self.num_features)
        count = 1
        for axis in axes:
            count *= x.shape[axis]
        if count > 1:
            var_data = var_data * (count / (count - 1))
        self.running_mean.data = (
            (1 - self.momentum) * self.running_mean.data
            + self.momentum * mean_data
        )
        self.running_var.data = (
            (1 - self.momentum) * self.running_var.data
            + self.momentum * var_data
        )

    def forward(self, x):
        x = ensure_tensor(x)
        self._check_input_dim(x)
        if x.shape[1] != self.num_features:
            raise ValueError(
                f'expected {self.num_features} features, got {x.shape[1]}'
            )

        axes = self._batch_axes(x)
        stat_shape = self._feature_shape(x.ndim)
        if self.training or not self.track_running_stats:
            mean = x.mean(axis=axes, keepdims=True)
            centered = x - mean
            var = (centered * centered).mean(axis=axes, keepdims=True)
            if self.training and self.track_running_stats:
                self._update_running_stats(x, mean, var, axes)
        else:
            mean = self.running_mean.reshape(stat_shape)
            var = self.running_var.reshape(stat_shape)
            centered = x - mean

        y = centered / ((var + self.eps) ** 0.5)
        if self.affine:
            y = y * self.weight.reshape(stat_shape) + self.bias.reshape(stat_shape)
        return y


class BatchNorm1d(_BatchNorm):
    def _check_input_dim(self, x):
        if x.ndim not in (2, 3):
            raise ValueError('BatchNorm1d expects 2D or 3D input')


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, x):
        if x.ndim != 4:
            raise ValueError('BatchNorm2d expects 4D input')
