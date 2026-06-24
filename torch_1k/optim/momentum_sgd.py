from .optimizer import Optimizer
from torch_1k import backend


class MomentumSGD(Optimizer):

    def __init__(self, parameters, lr=1e-2, momentum=0.9):
        super().__init__(parameters, defaults={'lr': lr, 'momentum': momentum})
        self._refresh_attributes_from_first_group()
        self.velocity = {}

    def _refresh_attributes_from_first_group(self):
        group = self.param_groups[0]
        self.lr = group.get('lr', getattr(self, 'lr', 1e-2))
        self.momentum = group.get('momentum', getattr(self, 'momentum', 0.9))

    def update_one(self, parameter, group):
        key = id(parameter)
        if key not in self.velocity:
            xp = backend.get_array_module(parameter.data)
            self.velocity[key] = xp.zeros_like(parameter.data)
        v = self.velocity[key]
        v = group['momentum'] * v - group['lr'] * parameter.grad.data
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
            'param_groups': self._state_param_groups(),
        }

    def load_state_dict(self, state_dict):
        self._load_param_groups(state_dict)
        self.velocity = {}
        for index, values in state_dict.get('state', {}).items():
            parameter = self.parameters[int(index)]
            self.velocity[id(parameter)] = self._state_array(
                values['velocity'], parameter
            )
