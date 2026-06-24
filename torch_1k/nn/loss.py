import numpy as np
from torch_1k.function import Function
from torch_1k.tensor import Tensor
from torch_1k import backend


class MSELoss(Function):

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def forward(self, input, target):
        self.input = input
        self.target = target
        # 所有元素的mean
        xp = backend.get_array_module(input)
        return xp.mean((input - target) ** 2)

    def backward(self, grad_output):
        # 从上下文中取出正向传播保存的变量
        input, target = self.input, self.target
        # 计算梯度
        grad_input = (2.0 / np.prod(input.shape)) * (input - target)
        # 返回输入的梯度，target的梯度为None
        return Tensor(grad_input) * grad_output


class CrossEntropyLoss(Function):
    def forward(self, input, target):
        xp = backend.get_array_module(input)
        target = target.astype("int64")
        self.target = target

        shifted = input - xp.max(input, axis=1, keepdims=True)
        exp_logits = xp.exp(shifted)
        self.probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)

        batch_index = xp.arange(input.shape[0])
        losses = -xp.log(self.probs[batch_index, target])
        return xp.mean(losses)

    def backward(self, grad_output):
        xp = backend.get_array_module(self.probs)
        grad = self.probs.copy()
        batch_index = xp.arange(self.target.shape[0])
        grad[batch_index, self.target] -= 1
        grad = grad / self.target.shape[0]
        return Tensor(grad) * grad_output
