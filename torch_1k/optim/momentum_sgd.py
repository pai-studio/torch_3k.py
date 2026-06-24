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

    def state_dict(self):
        indices = self._param_indices()
        state = {}
        for key, velocity in self.velocity.items():
            if key in indices:
                state[indices[key]] = {
                    'velocity': self._state_tensor(velocity),
                }
        return {
            'state': state,
            'param_groups': [
                self._param_group(lr=self.lr, momentum=self.momentum),
            ],
        }

    def load_state_dict(self, state_dict):
        group = self._validate_state_dict(state_dict)
        self.lr = group.get('lr', self.lr)
        self.momentum = group.get('momentum', self.momentum)
        self.velocity = {}
        for index, values in state_dict.get('state', {}).items():
            parameter = self.parameters[int(index)]
            self.velocity[id(parameter)] = self._state_array(
                values['velocity'], parameter
            )
