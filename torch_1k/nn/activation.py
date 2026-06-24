from .module import Module
from . import functional as F


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError('dropout probability has to be between 0 and 1')
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        return F.dropout(x, p=self.p, training=self.training,
                         inplace=self.inplace)


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.softmax(x, dim=self.dim)


class LogSoftmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.log_softmax(x, dim=self.dim)
