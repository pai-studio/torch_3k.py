from .module import Module


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        ndim = x.ndim
        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else ndim + self.end_dim
        shape = list(x.shape)
        flat = 1
        for dim in shape[start:end + 1]:
            flat *= dim
        return x.reshape(tuple(shape[:start] + [flat] + shape[end + 1:]))
