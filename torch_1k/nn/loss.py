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


def _array_scalar_bool(value):
    return bool(backend.as_numpy(value).item())


def _class_weight_data(weight, input_data, num_classes, dtype):
    if weight is None:
        return None

    data = weight.data if isinstance(weight, Tensor) else weight
    data = backend.to_device(data, backend.device_of(input_data))
    if data.shape != (num_classes,):
        raise ValueError(
            "cross_entropy weight must have shape (C,), "
            f"got {data.shape}"
        )
    return data.astype(dtype)


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
    def __init__(
        self, weight=None, ignore_index=-100, reduction="mean",
        label_smoothing=0.0,
    ):
        super().__init__()
        if label_smoothing < 0.0 or label_smoothing > 1.0:
            raise ValueError("label_smoothing must be between 0.0 and 1.0")
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = _check_reduction(reduction)
        self.label_smoothing = float(label_smoothing)

    def forward(self, input, target):
        if input.ndim < 2:
            raise ValueError("cross_entropy expects input shape (N, C, ...)")

        xp = backend.get_array_module(input)
        num_classes = input.shape[1]
        target = target.astype("int64")
        expected_target_shape = (input.shape[0],) + input.shape[2:]
        if target.shape != expected_target_shape:
            raise ValueError(
                "cross_entropy target shape must match input without "
                f"class dimension: expected {expected_target_shape}, "
                f"got {target.shape}"
            )

        if input.ndim == 2:
            logits = input.reshape((-1, num_classes))
        else:
            logits = xp.moveaxis(input, 1, -1).reshape((-1, num_classes))

        target_flat = target.reshape(-1)
        valid_mask = target_flat != self.ignore_index
        invalid_target = valid_mask & (
            (target_flat < 0) | (target_flat >= num_classes)
        )
        if _array_scalar_bool(xp.any(invalid_target)):
            raise ValueError(
                "cross_entropy target values must be in [0, C) or "
                "equal to ignore_index"
            )

        safe_target = target_flat.copy()
        safe_target = xp.where(valid_mask, safe_target, xp.zeros_like(safe_target))

        shifted = logits - xp.max(logits, axis=1, keepdims=True)
        exp_logits = xp.exp(shifted)
        self.probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)
        log_probs = shifted - xp.log(xp.sum(exp_logits, axis=1, keepdims=True))

        weight_data = _class_weight_data(
            self.weight, input, num_classes, log_probs.dtype,
        )
        if weight_data is None:
            sample_weight = xp.ones(target_flat.shape, dtype=log_probs.dtype)
        else:
            sample_weight = weight_data[safe_target]
        sample_weight = xp.where(
            valid_mask, sample_weight, xp.zeros_like(sample_weight),
        )

        batch_index = xp.arange(target_flat.shape[0])
        nll = -log_probs[batch_index, safe_target]
        if weight_data is not None:
            nll = nll * sample_weight

        if self.label_smoothing:
            if weight_data is None:
                smooth_loss = -xp.sum(log_probs, axis=1)
            else:
                smooth_loss = -xp.sum(
                    log_probs * weight_data.reshape((1, num_classes)),
                    axis=1,
                )
            losses = (
                (1.0 - self.label_smoothing) * nll
                + (self.label_smoothing / num_classes) * smooth_loss
            )
        else:
            losses = nll
        losses = xp.where(valid_mask, losses, xp.zeros_like(losses))

        self.input_shape = input.shape
        self.target_shape = target.shape
        self.num_classes = num_classes
        self.target = target_flat
        self.safe_target = safe_target
        self.valid_mask = valid_mask
        self.weight_data = weight_data
        self.sample_weight = sample_weight
        self.mean_normalizer = None
        self.mean_normalizer_nonzero = True

        if self.reduction == "none":
            return losses.reshape(target.shape)
        if self.reduction == "sum":
            return xp.sum(losses)

        if weight_data is None:
            normalizer = xp.sum(valid_mask.astype(log_probs.dtype))
        else:
            normalizer = xp.sum(sample_weight)
        self.mean_normalizer = normalizer
        self.mean_normalizer_nonzero = _array_scalar_bool(normalizer != 0)
        return xp.sum(losses) / normalizer

    def backward(self, grad_output):
        xp = backend.get_array_module(self.probs)
        batch_index = xp.arange(self.target.shape[0])

        if self.label_smoothing:
            smooth = self.label_smoothing / self.num_classes
            target_dist = xp.full(self.probs.shape, smooth,
                                  dtype=self.probs.dtype)
            target_dist[batch_index, self.safe_target] += (
                1.0 - self.label_smoothing
            )
            if self.weight_data is None:
                grad = self.probs - target_dist
            else:
                target_dist = target_dist * self.weight_data.reshape(
                    (1, self.num_classes)
                )
                grad = (
                    self.probs * xp.sum(target_dist, axis=1, keepdims=True)
                    - target_dist
                )
        else:
            grad = self.probs.copy()
            grad[batch_index, self.safe_target] -= 1.0
            if self.weight_data is not None:
                grad = grad * self.sample_weight.reshape((-1, 1))

        grad = xp.where(
            self.valid_mask.reshape((-1, 1)), grad, xp.zeros_like(grad),
        )
        if self.reduction == "mean":
            if self.mean_normalizer_nonzero:
                grad = grad / self.mean_normalizer
            else:
                grad = xp.zeros_like(grad)
        elif self.reduction == "none":
            grad = grad * grad_output.data.reshape((-1, 1))

        if self.reduction != "none":
            grad = grad * grad_output.data

        if len(self.input_shape) == 2:
            grad_input = grad.reshape(self.input_shape)
        else:
            grad_input = grad.reshape(self.target_shape + (self.num_classes,))
            grad_input = xp.moveaxis(grad_input, -1, 1)
        return Tensor(grad_input), None
