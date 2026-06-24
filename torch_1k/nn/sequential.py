from .module import Module


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._modules = []
        for index, module in enumerate(modules):
            self.add_module(str(index), module)

    def add_module(self, name, module):
        self._modules.append(name)
        setattr(self, name, module)

    def forward(self, x):
        for name in self._modules:
            x = getattr(self, name)(x)
        return x
