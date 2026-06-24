import weakref
from .parameter import Parameter
from ..tensor import Tensor
from .. import backend
from ..settings import train_model, eval_model


class Module:

    def __init__(self):
        self._parameters = []
        self.training = True

    def train(self, mode=True):
        self.training = mode
        if mode:
            train_model()
        else:
            eval_model()
        for name in self._parameters:
            obj = self.__dict__[name]
            if isinstance(obj, Module):
                obj.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def to(self, device):
        for name in self._parameters:
            obj = self.__dict__[name]
            if isinstance(obj, Module):
                obj.to(device)
            elif isinstance(obj, Parameter):
                obj.data = backend.to_device(obj.data, device)
                if obj.grad is not None:
                    obj.grad = obj.grad.to(device)
        return self

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Module)):
            names = self.__dict__.get('_parameters')
            if names is not None and name not in names:
                names.append(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        self.inputs = [weakref.ref(input) for input in inputs]
        self.outputs = [weakref.ref(output) for output in outputs]
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, x):
        raise NotImplementedError()

    def parameters(self):
        for _, parameter in self.named_parameters():
            yield parameter

    def named_parameters(self, prefix=''):
        for name in self._parameters:
            obj = self.__dict__[name]
            # support Nested-Module
            if isinstance(obj, Module):
                child_prefix = f'{prefix}{name}.'
                yield from obj.named_parameters(child_prefix)
            else:
                yield f'{prefix}{name}', obj

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()

    def state_dict(self, prefix=''):
        state = {}
        for name in self._parameters:
            obj = self.__dict__[name]
            key = f'{prefix}{name}'
            if isinstance(obj, Module):
                state.update(obj.state_dict(prefix=f'{key}.'))
            else:
                state[key] = Tensor(obj.data.copy())
        return state

    def load_state_dict(self, state_dict, strict=True, prefix=''):
        expected = dict(self.named_parameters(prefix))
        missing = []
        for key, parameter in expected.items():
            if key not in state_dict:
                missing.append(key)
                continue
            value = state_dict[key]
            if isinstance(value, Tensor):
                value = value.data
            data = backend.to_device(value, parameter.device)
            parameter.data = data.copy()

        unexpected = [
            key for key in state_dict
            if key.startswith(prefix) and key not in expected
        ]
        if strict and (missing or unexpected):
            raise KeyError(
                f'state_dict mismatch: missing={missing}, unexpected={unexpected}'
            )
        return missing
