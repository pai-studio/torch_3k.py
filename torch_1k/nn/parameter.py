from ..tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data, name=None, log_enabled=None, device=None, dtype=None,
                 requires_grad=True):
        super().__init__(
            data, name=name, log_enabled=log_enabled, device=device,
            dtype=dtype, requires_grad=requires_grad,
        )
