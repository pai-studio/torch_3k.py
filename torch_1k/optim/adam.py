from .optimizer import Optimizer
from torch_1k import backend


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        super().step()

    def update_one(self, parameter):
        key = id(parameter)
        xp = backend.get_array_module(parameter.data)
        if key not in self.m:
            self.m[key] = xp.zeros_like(parameter.data)
            self.v[key] = xp.zeros_like(parameter.data)

        grad = parameter.grad.data
        if self.weight_decay:
            grad = grad + self.weight_decay * parameter.data

        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad * grad)

        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        parameter.data = parameter.data - self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2):
        super().__init__(parameters, lr=lr, betas=betas, eps=eps,
                         weight_decay=0.0)
        self.decoupled_weight_decay = weight_decay

    def update_one(self, parameter):
        if self.decoupled_weight_decay:
            parameter.data = parameter.data * (1 - self.lr * self.decoupled_weight_decay)
        super().update_one(parameter)
