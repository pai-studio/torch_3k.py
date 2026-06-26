from torch_1k import backend
from torch_1k.tensor import Tensor
from torch_1k.functional.numeric import (
    ReLU, relu, softmax, log_softmax,
)
from torch_1k.nn.loss import CrossEntropyLoss, MSELoss


def dropout(input, p=0.5, training=True, inplace=False):
    if p < 0 or p > 1:
        raise ValueError('dropout probability has to be between 0 and 1')
    if not training or p == 0:
        return input
    if p == 1:
        return input * 0.0

    xp = backend.get_array_module(input.data)
    keep_prob = 1.0 - p
    mask = (xp.random.rand(*input.shape) < keep_prob).astype(input.data.dtype)
    mask = mask / keep_prob
    return input * Tensor(mask, requires_grad=False)


def cross_entropy(
    input, target, weight=None, ignore_index=-100, reduction="mean",
    label_smoothing=0.0,
):
    return CrossEntropyLoss(
        weight=weight,
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )(input, target)


def mse_loss(input, target, reduction="mean"):
    return MSELoss(reduction=reduction)(input, target)
