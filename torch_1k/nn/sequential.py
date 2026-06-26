from .module import Module


class Identity(Module):
    def forward(self, x):
        return x


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


class ModuleList(Module):
    def __init__(self, modules=None):
        super().__init__()
        self._module_names = []
        if modules is not None:
            self.extend(modules)

    def append(self, module):
        self.add_module(str(len(self._module_names)), module)
        return self

    def extend(self, modules):
        for module in modules:
            self.append(module)
        return self

    def add_module(self, name, module):
        if not isinstance(module, Module):
            raise TypeError('ModuleList only accepts Module instances')
        if name not in self._module_names:
            self._module_names.append(name)
        setattr(self, name, module)

    def __iter__(self):
        for name in self._module_names:
            yield getattr(self, name)

    def __len__(self):
        return len(self._module_names)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ModuleList(list(self)[index])
        if index < 0:
            index += len(self._module_names)
        return getattr(self, self._module_names[index])
