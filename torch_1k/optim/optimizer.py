from torch_1k import backend
from torch_1k.tensor import Tensor


class Optimizer:

    def __init__(self, parameters, defaults=None):
        self.defaults = defaults or {}
        self.parameters = []
        self.param_groups = []
        self._init_param_groups(parameters)

    def zero_grad(self, set_to_none=False):
        for parameter in self.parameters:
            if parameter.grad is not None:
                if set_to_none:
                    parameter.grad = None
                else:
                    parameter.grad.data.fill(0)

    def step(self):
        # print(f'{self.parameters=}')
        for group in self.param_groups:
            for parameter in group['params']:
                # print(parameter.grad)
                if parameter.grad is not None:
                    self.update_one(parameter, group)

    def update_one(self, parameter, group):
        raise NotImplementedError()

    def _init_param_groups(self, parameters):
        if isinstance(parameters, dict):
            groups = [parameters]
        else:
            groups = list(parameters)
        if groups and isinstance(groups[0], dict):
            for group in groups:
                self.add_param_group(group)
        else:
            self.add_param_group({'params': groups})

    def _param_list(self, params):
        if isinstance(params, Tensor):
            return [params]
        return [parameter for parameter in params]

    def add_param_group(self, group):
        params = self._param_list(group['params'])
        options = dict(self.defaults)
        for key, value in group.items():
            if key != 'params':
                options[key] = value
        options['params'] = params
        self.param_groups.append(options)
        self.parameters.extend(params)

    def _param_indices(self):
        return {
            id(parameter): index
            for index, parameter in enumerate(self.parameters)
        }

    def _state_param_groups(self):
        indices = self._param_indices()
        groups = []
        for group in self.param_groups:
            state_group = {
                key: value for key, value in group.items()
                if key != 'params'
            }
            state_group['params'] = [indices[id(p)] for p in group['params']]
            groups.append(state_group)
        return groups

    def _validate_state_dict(self, state_dict):
        loaded_groups = state_dict.get('param_groups', [])
        if len(loaded_groups) != len(self.param_groups):
            raise ValueError('loaded state dict has a different number of param groups')
        for loaded_group, group in zip(loaded_groups, self.param_groups):
            if len(loaded_group.get('params', [])) != len(group['params']):
                raise ValueError('loaded state dict has a different number of parameters')
        return loaded_groups

    def _load_param_groups(self, state_dict):
        loaded_groups = self._validate_state_dict(state_dict)
        for group, loaded_group in zip(self.param_groups, loaded_groups):
            for key, value in loaded_group.items():
                if key != 'params':
                    group[key] = value
        self._refresh_attributes_from_first_group()
        return loaded_groups

    def _refresh_attributes_from_first_group(self):
        pass

    def _state_tensor(self, value):
        if isinstance(value, Tensor):
            value = value.data
        return Tensor(value.copy(), requires_grad=False)

    def _state_array(self, value, parameter):
        if isinstance(value, Tensor):
            value = value.data
        return backend.to_device(value, parameter.device).copy()

    def state_dict(self):
        return {
            'state': {},
            'param_groups': self._state_param_groups(),
        }

    def load_state_dict(self, state_dict):
        self._load_param_groups(state_dict)
