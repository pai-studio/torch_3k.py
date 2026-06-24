from torch_1k import backend
from ..function import Function
from .matrix import sum_to


def _adjust_grad_shape(gx, shape):
    if gx.shape != shape:
        return sum_to(gx, shape)
    return gx


# Square
class Square(Function):

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # x = self.inputs[0].data
        x = self.inputs[0]
        return 2 * x * gy

def square(x):
    return Square()(x)


# Exp
class Exp(Function):

    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        # 为了计算高阶导数, 要求grad也为Tensor类型
        # x = self.inputs[0].data
        x = self.inputs[0]
        # return np.exp(x) * gy
        return exp(x) * gy

def exp(x):
    return Exp()(x)

# Neg
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

# Add
class Add(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        # 发生了隐式broadcast -> sum
        y = x1 + x2
        # self.broadcast_shape = y.shape
        return y

    def backward(self, gy):
        gx1, gx2 = gy, gy
        #print(f'***{self.x1_shape=}, {self.x2_shape=}, {self.broadcast_shape=}')
        #if self.x1_shape != self.broadcast_shape:
        #    gx1 = sum_to(gx1, self.x1_shape)
        #if self.x2_shape != self.broadcast_shape:
        #    # 梯度回到自己原来的shape
        #    gx2 = sum_to(gx2, self.x2_shape)
        # 与上面等价
        if self.x1_shape != self.x2_shape:
            x1, x2 = self.inputs
            gx1 = sum_to(gx1, self.x1_shape)
            gx2 = sum_to(gx2, self.x2_shape)
        return gx1, gx2

def add(x1, x2):
    return Add()(x1, x2)


# Sub
class Sub(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        return x1 - x2

    def backward(self, gy):
        gx1 = _adjust_grad_shape(gy, self.x1_shape)
        gx2 = _adjust_grad_shape(-gy, self.x2_shape)
        return gx1, gx2

def sub(x1, x2):
    return Sub()(x1, x2)

def rsub(x1, x2):
    # swap
    return Sub()(x2, x1)


# Mul
class Mul(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        return x1 * x2

    def backward(self, gy):
        x1, x2 = self.inputs[0], self.inputs[1]
        gx1 = _adjust_grad_shape(x2*gy, self.x1_shape)
        gx2 = _adjust_grad_shape(x1*gy, self.x2_shape)
        return gx1, gx2

def mul(x1, x2):
    return Mul()(x1, x2)

# Div
class Div(Function):
    def forward(self, x1, x2):
        self.x1_shape, self.x2_shape = x1.shape, x2.shape
        return x1 / x2

    def backward(self, gy):
        x1, x2 = self.inputs[0], self.inputs[1]
        gx1 = _adjust_grad_shape(gy/x2, self.x1_shape)
        gx2 = _adjust_grad_shape(-gy*x1 / x2 ** 2, self.x2_shape)
        return gx1, gx2

def div(x1, x2):
    return Div()(x1, x2)

def rdiv(x1, x2):
    # swap
    return Div()(x2, x1)

# Pow
class Pow(Function):
    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        gx = c * x **(c-1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

# Sin
class Sin(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.sin(x)

    def backward(self, gy):
        x = self.inputs[0]
        # gx = gy * np.cos(x)
        gx = gy * cos(x) # call Cos()(x)
        return gx

def sin(x):
    return Sin()(x)

# Cos
class Cos(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.cos(x)

    def backward(self, gy):
        x = self.inputs[0]
        # gx = gy * np.sin(x)
        gx = - gy * sin(x)
        return gx

def cos(x):
    return Cos()(x)

# Tanh
class Tanh(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return xp.tanh(x)

    def backward(self, gy):
        # 1 - y**2
        y = self.outputs[0]()
        gx = gy * (1 - y*y)
        return gx

def tanh(x):
    return Tanh()(x)

# Sigmoid
class Sigmoid(Function):
    def forward(self, x):
        xp = backend.get_array_module(x)
        return 1/(1+xp.exp(-x))

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * y * (1 - y)

def sigmoid(x):
    return Sigmoid()(x)
