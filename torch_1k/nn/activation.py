from .module import Module
from . import functional as F


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)
