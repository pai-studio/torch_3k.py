from torch_1k import backend
from torch_1k.tensor import Tensor


class Optimizer:

    def __init__(self, parameters):
        self.parameters = [item for item in parameters]
        # print([item for item in self.parameters])

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                # parameter.grad = None
                parameter.grad.data.fill(0)

    def step(self):
        # print(f'{self.parameters=}')
        for parameter in self.parameters:
            # print(parameter.grad)
            if parameter.grad is not None:
                self.update_one(parameter)

    def update_one(self, parameter):
        raise NotImplementedError()

    def _param_indices(self):
        return {
            id(parameter): index
            for index, parameter in enumerate(self.parameters)
        }

    def _param_group(self, **options):
        group = {'params': list(range(len(self.parameters)))}
        group.update(options)
        return group

    def _validate_state_dict(self, state_dict):
        param_groups = state_dict.get('param_groups', [])
        if len(param_groups) != 1:
            raise ValueError('expected exactly one optimizer param group')
        group = param_groups[0]
        if len(group.get('params', [])) != len(self.parameters):
            raise ValueError('loaded state dict has a different number of parameters')
        return group

    def _state_tensor(self, value):
        if isinstance(value, Tensor):
            value = value.data
        return Tensor(value.copy())

    def _state_array(self, value, parameter):
        if isinstance(value, Tensor):
            value = value.data
        return backend.to_device(value, parameter.device).copy()

    def state_dict(self):
        return {
            'state': {},
            'param_groups': [self._param_group()],
        }

    def load_state_dict(self, state_dict):
        self._validate_state_dict(state_dict)
