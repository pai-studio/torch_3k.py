import weakref
from .parameter import Parameter
from ..tensor import Tensor, _parse_to_args
from .. import backend
from ..settings import train_model, eval_model


class Module:

    def __init__(self):
        self._parameters = []
        self._buffers = []
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

    def to(self, *args, device=None, dtype=None):
        device, dtype = _parse_to_args(args, device, dtype)
        for name in self._parameters:
            obj = self.__dict__[name]
            if isinstance(obj, Module):
                obj.to(device=device, dtype=dtype)
            elif isinstance(obj, Parameter):
                obj.data = backend.to_device(obj.data, device, dtype=dtype)
                if obj.grad is not None:
                    obj.grad = obj.grad.to(device=device, dtype=dtype)
        for name in self._buffers:
            obj = self.__dict__[name]
            obj.data = backend.to_device(obj.data, device, dtype=dtype)
            if obj.grad is not None:
                obj.grad = obj.grad.to(device=device, dtype=dtype)
        return self

    def __setattr__(self, name, value):
        parameters = self.__dict__.get('_parameters')
        buffers = self.__dict__.get('_buffers')
        if isinstance(value, (Parameter, Module)):
            if parameters is not None and name not in parameters:
                parameters.append(name)
            if buffers is not None and name in buffers:
                buffers.remove(name)
        elif isinstance(value, Tensor):
            if buffers is not None and name not in buffers:
                buffers.append(name)
            if parameters is not None and name in parameters:
                parameters.remove(name)
        else:
            if parameters is not None and name in parameters:
                parameters.remove(name)
            if buffers is not None and name in buffers:
                buffers.remove(name)
        super().__setattr__(name, value)

    def register_buffer(self, name, tensor):
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor, requires_grad=False)
        tensor.requires_grad = False
        setattr(self, name, tensor)
        return tensor

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

    def children(self):
        for _, module in self.named_children():
            yield module

    def named_children(self):
        seen = set()
        for name in self._parameters:
            obj = self.__dict__[name]
            if isinstance(obj, Module) and id(obj) not in seen:
                seen.add(id(obj))
                yield name, obj

    def modules(self):
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for name, module in self.named_children():
            child_prefix = f'{prefix}.{name}' if prefix else name
            yield from module.named_modules(memo, child_prefix)

    def named_parameters(self, prefix=''):
        for name in self._parameters:
            obj = self.__dict__[name]
            # support Nested-Module
            if isinstance(obj, Module):
                child_prefix = f'{prefix}{name}.'
                yield from obj.named_parameters(child_prefix)
            else:
                yield f'{prefix}{name}', obj

    def named_buffers(self, prefix=''):
        for name in self._buffers:
            yield f'{prefix}{name}', self.__dict__[name]
        for name in self._parameters:
            obj = self.__dict__[name]
            if isinstance(obj, Module):
                child_prefix = f'{prefix}{name}.'
                yield from obj.named_buffers(child_prefix)

    def zero_grad(self, set_to_none=False):
        for parameter in self.parameters():
            if set_to_none or parameter.grad is None:
                parameter.zero_grad()
            else:
                parameter.grad.data.fill(0)

    def state_dict(self, prefix=''):
        state = {}
        for name in self._parameters:
            obj = self.__dict__[name]
            key = f'{prefix}{name}'
            if isinstance(obj, Module):
                state.update(obj.state_dict(prefix=f'{key}.'))
            else:
                state[key] = Tensor(obj.data.copy(), requires_grad=False)
        for name in self._buffers:
            obj = self.__dict__[name]
            state[f'{prefix}{name}'] = Tensor(obj.data.copy(), requires_grad=False)
        return state

    def load_state_dict(self, state_dict, strict=True, prefix=''):
        expected = dict(self.named_parameters(prefix))
        expected.update(dict(self.named_buffers(prefix)))
        missing = []
        for key, tensor in expected.items():
            if key not in state_dict:
                missing.append(key)
                continue
            value = state_dict[key]
            if isinstance(value, Tensor):
                value = value.data
            data = backend.to_device(value, tensor.device)
            tensor.data = data.copy()

        unexpected = [
            key for key in state_dict
            if key.startswith(prefix) and key not in expected
        ]
        if strict and (missing or unexpected):
            raise KeyError(
                f'state_dict mismatch: missing={missing}, unexpected={unexpected}'
            )
        return missing
