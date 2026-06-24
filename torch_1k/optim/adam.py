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

    def _param_group_options(self):
        return {
            'lr': self.lr,
            'betas': (self.beta1, self.beta2),
            'eps': self.eps,
            'weight_decay': self.weight_decay,
        }

    def _load_param_group_options(self, group):
        self.lr = group.get('lr', self.lr)
        betas = group.get('betas', (self.beta1, self.beta2))
        self.beta1, self.beta2 = betas
        self.eps = group.get('eps', self.eps)
        self.weight_decay = group.get('weight_decay', self.weight_decay)

    def state_dict(self):
        indices = self._param_indices()
        state = {}
        for key, m_value in self.m.items():
            if key in indices:
                state[indices[key]] = {
                    'm': self._state_tensor(m_value),
                    'v': self._state_tensor(self.v[key]),
                }
        return {
            'state': state,
            'param_groups': [self._param_group(**self._param_group_options())],
            't': self.t,
        }

    def load_state_dict(self, state_dict):
        group = self._validate_state_dict(state_dict)
        self._load_param_group_options(group)
        self.t = state_dict.get('t', self.t)
        self.m = {}
        self.v = {}
        for index, values in state_dict.get('state', {}).items():
            parameter = self.parameters[int(index)]
            key = id(parameter)
            self.m[key] = self._state_array(values['m'], parameter)
            self.v[key] = self._state_array(values['v'], parameter)


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

    def _param_group_options(self):
        options = super()._param_group_options()
        options['weight_decay'] = self.decoupled_weight_decay
        return options

    def _load_param_group_options(self, group):
        self.lr = group.get('lr', self.lr)
        betas = group.get('betas', (self.beta1, self.beta2))
        self.beta1, self.beta2 = betas
        self.eps = group.get('eps', self.eps)
        self.decoupled_weight_decay = group.get(
            'weight_decay', self.decoupled_weight_decay
        )
        self.weight_decay = 0.0
