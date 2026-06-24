from .optimizer import Optimizer
from torch_1k import backend


class MomentumSGD(Optimizer):

    def __init__(self, parameters, lr=1e-2, momentum=0.9):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

    def update_one(self, parameter):
        key = id(parameter)
        if key not in self.velocity:
            xp = backend.get_array_module(parameter.data)
            self.velocity[key] = xp.zeros_like(parameter.data)
        v = self.velocity[key]
        v = self.momentum * v - self.lr * parameter.grad.data
        parameter.data = parameter.data + v
        self.velocity[key] = v
