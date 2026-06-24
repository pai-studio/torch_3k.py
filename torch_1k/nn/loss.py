import numpy as np
from torch_1k.function import Function
from torch_1k.tensor import Tensor
from torch_1k import backend


def _check_reduction(reduction):
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(
            "reduction must be one of 'none', 'mean', or 'sum'"
        )
    return reduction


def _sum_to_input_shape(grad, shape):
    if grad is None or grad.shape == shape:
        return grad
    from torch_1k.functional.matrix import sum_to

    return sum_to(grad, shape)


class MSELoss(Function):

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = _check_reduction(reduction)

    def forward(self, input, target):
        self.input = input
        self.target = target
        xp = backend.get_array_module(input)
        loss = (input - target) ** 2
        if self.reduction == "none":
            return loss
        if self.reduction == "sum":
            return xp.sum(loss)
        return xp.mean(loss)

    def backward(self, grad_output):
        input, target = self.input, self.target
        grad_input = Tensor(2.0 * (input - target), requires_grad=False)
        if self.reduction == "mean":
            grad_input = grad_input / np.prod((input - target).shape)
        grad_input = grad_input * grad_output
        grad_target = -grad_input
        return (
            _sum_to_input_shape(grad_input, input.shape),
            _sum_to_input_shape(grad_target, target.shape),
        )


class CrossEntropyLoss(Function):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = _check_reduction(reduction)

    def forward(self, input, target):
        if input.ndim != 2:
            raise ValueError("cross_entropy expects input shape (N, C)")

        xp = backend.get_array_module(input)
        target = target.astype("int64").reshape(-1)
        if target.shape[0] != input.shape[0]:
            raise ValueError("cross_entropy target length must match batch size")

        self.target = target
        self.batch_size = input.shape[0]

        shifted = input - xp.max(input, axis=1, keepdims=True)
        exp_logits = xp.exp(shifted)
        self.probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)

        batch_index = xp.arange(input.shape[0])
        losses = -xp.log(self.probs[batch_index, target])
        if self.reduction == "none":
            return losses
        if self.reduction == "sum":
            return xp.sum(losses)
        return xp.mean(losses)

    def backward(self, grad_output):
        xp = backend.get_array_module(self.probs)
        grad = self.probs.copy()
        batch_index = xp.arange(self.target.shape[0])
        grad[batch_index, self.target] -= 1
        grad = Tensor(grad)
        if self.reduction == "mean":
            grad = grad / self.batch_size
        elif self.reduction == "none":
            grad_output = grad_output.reshape((-1, 1))
        return grad * grad_output, None
