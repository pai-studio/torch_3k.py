# torch_1k 核心原理说明

日期：2026-06-24

源码基准 Git 记录：`main@93e9acc0bd7538fdf653db747ff6d66546acb515`

说明：本文档基于上述 Git 记录中的 `torch_1k` 源码阅读生成；文档自身为该工作区后续新增内容。

## 1. 项目定位

`torch_1k` 的目标是在约 1000 行核心代码内复刻 PyTorch 最核心的一条工作流：

1. 用 `Tensor` 保存数值和梯度。
2. 用可组合算子执行前向计算。
3. 在前向计算时动态构建计算图。
4. 从标量损失反向传播梯度。
5. 用 `Module`、`Parameter`、`Linear`、`MSELoss`、`SGD` 组成一个最小神经网络训练闭环。

它不是完整 PyTorch 的轻量实现，而是一个教学型自动微分框架。源码主要借鉴 DeZero 风格：每个算子是一个 `Function`，每个计算结果是一个 `Tensor`，`Tensor.creator` 指向产生它的 `Function`，反向传播沿这些指针反向遍历。

## 2. 源码阅读范围

本说明基于当前仓库中的以下文件：

- `torch_1k/tensor.py`：`Tensor` 数据结构、反向传播调度、操作符注册。
- `torch_1k/function.py`：所有可微算子的基类协议。
- `torch_1k/functional/numeric.py`：逐元素数值算子。
- `torch_1k/functional/matrix.py`：形状变换、矩阵乘法、广播、求和、线性层底层函数。
- `torch_1k/functional/get_item.py`、`torch_1k/functional/pad.py`：切片和切片梯度回填。
- `torch_1k/nn/*.py`：`Module`、`Parameter`、`Linear`、`MSELoss`。
- `torch_1k/optim/*.py`：优化器基类和 `SGD`。
- `torch_1k/settings.py`：全局运行开关。
- `torch_1k/misc.py`、`torch_1k/utils/*.py`：PyTorch-like 辅助 API 和 NumPy 工具。
- `torch_1k/utils/data/*.py`：最小数据集与数据加载器雏形。

## 3. 总体架构

源码可以分成五层：

```text
用户代码
  |
  v
Tensor API / torch_1k 顶层 API
  |
  v
Function 子类：Add、Mul、MatMul、Sum、MSELoss ...
  |
  v
NumPy：实际数值计算
  |
  v
反向传播：Tensor.backward() 按动态图反向调用 Function.backward()
```

核心原则是：前向计算尽量交给 NumPy，自动微分逻辑由 `Tensor` 和 `Function` 维护。这样每个算子只需要实现两件事：

- `forward(...)`：输入 NumPy 数组，输出 NumPy 数组。
- `backward(...)`：输入输出端梯度，返回输入端梯度。

## 4. Tensor：数值、梯度和图节点

`Tensor` 是框架的基本数据结构，主要字段如下：

- `data`：由 `utils.ensure_ndarray` 转成的 `np.ndarray`。
- `grad`：反向传播得到的梯度，通常也是一个 `Tensor`。
- `creator`：产生当前 Tensor 的 `Function`。
- `generation`：当前 Tensor 在计算图中的代数，用于反向传播排序。
- `name`：调试用名称。

`Tensor` 不直接实现大量计算逻辑，而是在 `register_ops()` 中把 Python 魔术方法绑定到 `functional` 层：

- `+` / `-` / `*` / `/` / `**` 分别绑定到 `F.add`、`F.sub`、`F.mul`、`F.div`、`F.pow`。
- `Tensor.__getitem__` 绑定到 `get_item`。
- `Tensor.T`、`reshape()`、`sum()` 等方法转调到 `functional.matrix`。

这种设计让用户写法接近 PyTorch：

```python
y = ((x * W).sum() + b) ** 2
y.backward()
```

但底层每一步都会生成一个 `Function` 节点和输出 `Tensor`。

## 5. Function：动态图构建协议

`Function.__call__` 是整个框架最关键的入口，它完成以下流程：

1. 把所有输入通过 `ensure_tensor` 转成 `Tensor`。
2. 取出每个输入的 `.data`，调用子类 `forward` 做 NumPy 前向计算。
3. 把 `forward` 的返回值重新包装成 `Tensor`。
4. 如果 `Config.enable_backprop` 为真，且至少一个输入 `requires_grad=True`，就把输出 `Tensor` 的 `creator` 设为当前 `Function`。
5. 保存当前算子的 `inputs`，并用弱引用保存 `outputs`。
6. 返回单个输出或输出元组。

动态图连接关系如下：

```text
x ----\
       Add() ---> y
x ----/

y.creator = Add()
Add.inputs = [x, x]
Add.outputs = [weakref(y)]
```

这意味着计算图不是预先定义的，而是在执行 Python 前向代码时即时生成。控制流、普通 Python 函数和复合表达式天然可以参与建图，只要它们最终调用的是 `Function` 子类。

## 6. 反向传播调度

`Tensor.backward()` 实现的是反向模式自动微分。对最终输出 `y` 调用 `y.backward()` 后，流程是：

1. 如果 `y.grad is None`，用 `ones_like(y.data)` 初始化输出梯度。
2. 从 `y.creator` 开始，把待处理函数放入列表。
3. 每次取 `generation` 最大的函数，保证更靠后的节点先反传。
4. 从函数的输出 Tensor 上取 `grad`，调用该函数的 `backward`。
5. 把返回的梯度累加到每个输入 Tensor 的 `.grad`。
6. 如果输入 Tensor 也有 `creator`，继续加入待处理列表。
7. 默认清掉中间输出的梯度，降低内存占用。

`generation` 解决了复杂图中的拓扑顺序问题。例如：

```python
a = square(x)
y = add(square(a), square(a))
```

同一个中间变量 `a` 被两个分支复用。反传时必须先处理 `y` 附近的节点，再回到 `a`，最后回到 `x`。`generation` 加上去重集合可以避免顺序错误和重复入队。

梯度累加逻辑也处理了 `x + x` 这类同一变量多次出现的情况：

```python
if x.grad is None:
    x.grad = gx
else:
    x.grad = x.grad + gx
```

因此 `add(x, x).backward()` 能得到 `x.grad = 2`。

## 7. 高阶导数机制

`backward(create_graph=True)` 是高阶导数的关键。默认反向传播时不会为梯度计算继续建图；当 `create_graph=True` 时，`Tensor.backward()` 会临时设置：

```python
with using_config('enable_backprop', create_graph):
    gxs = f.backward(*gys)
```

如果算子的 `backward` 内部使用的是 `Tensor` 运算，而不是纯 NumPy 运算，那么梯度表达式本身也会形成新的计算图。典型例子：

- `Square.backward` 使用 `2 * x * gy`，其中 `x` 是原始输入 `Tensor`。
- `Exp.backward` 使用 `exp(x) * gy`。
- `Sin.backward` 使用 `cos(x)`。
- `Cos.backward` 使用 `-gy * sin(x)`。

这使得如下流程可行：

```python
y = x ** 3
y.backward(create_graph=True)
gx = x.grad
x.zero_grad()
gx.backward()
```

第一次反传得到的一阶导数仍然是可微的，第二次反传即可得到二阶导数。

## 8. 内存管理：弱引用和中间梯度释放

计算图天然有循环引用：

```text
Tensor(output) -> creator(Function) -> outputs -> Tensor(output)
```

源码通过两个策略减小内存压力：

1. `Function.outputs` 默认保存 `weakref.ref(output)`，由 `runtime_settings['remove_recursive_ref']` 控制。
2. `Tensor.backward(retain_grad=False)` 默认在每个函数反传完成后，把中间输出 Tensor 的 `.grad` 清空。

这样更接近训练时的需求：用户通常只关心叶子参数的梯度，而不关心每个中间结果的梯度。

## 9. 逐元素算子

`functional/numeric.py` 中的基础算子都继承自 `Function`：

- `Square`：前向 `x ** 2`，反向 `2 * x * gy`。
- `Exp`：前向 `np.exp(x)`，反向 `exp(x) * gy`。
- `Neg`：反向 `-gy`。
- `Add`：反向把同一个梯度传给两个输入，必要时通过 `sum_to` 还原广播前形状。
- `Sub`：反向为 `(gy, -gy)`。
- `Mul`：反向为 `(x2 * gy, x1 * gy)`。
- `Div`：反向为 `(gy / x2, -gy * x1 / x2 ** 2)`。
- `Pow`：反向为 `c * x ** (c - 1) * gy`。
- `Sin` / `Cos` / `Tanh` / `Sigmoid`：展示了非线性函数的扩展方式。

这些算子体现了一个统一扩展模式：

```python
class NewOp(Function):
    def forward(self, x):
        return numpy_forward(x)

    def backward(self, gy):
        x = self.inputs[0]
        return tensor_backward_formula(x, gy)

def new_op(x):
    return NewOp()(x)
```

只要 `backward` 中使用 `Tensor` 级别的表达式，就可以支持高阶导数。

## 10. 广播与形状还原

NumPy 的广播会在前向中自动扩展形状，例如：

```python
x.shape == (2, 3)
b.shape == (1, 3)
y = x + b
```

反向传播时，`b` 的梯度不能保持 `(2, 3)`，必须沿被广播的维度求和回 `(1, 3)`。`torch_1k` 用两组函数处理这个问题：

- `broadcast_to(x, shape)`：前向扩展形状，反向调用 `sum_to`。
- `sum_to(x, shape)`：前向缩减到目标形状，反向调用 `broadcast_to`。

`Add.backward` 会记录两个输入的原始形状，当形状不同就分别调用 `sum_to`：

```python
if self.x1_shape != self.x2_shape:
    gx1 = sum_to(gx1, self.x1_shape)
    gx2 = sum_to(gx2, self.x2_shape)
```

当前广播修正主要覆盖 `Add`、`Broadcast`、`SumTo` 这条路径。`Mul`、`Div` 等二元算子的广播梯度还没有做同样的形状还原，这是后续补全 PyTorch 兼容性时需要处理的重点。

## 11. 矩阵、求和和线性层底座

`functional/matrix.py` 实现了神经网络需要的核心形状算子：

- `Reshape`：前向 reshape，反向 reshape 回原始形状。
- `Transpose`：前向转置，反向再次转置。
- `MatMul`：前向 `x.dot(W)`，反向：
  - `gx = gy @ W.T`
  - `gW = x.T @ gy`
- `Sum`：前向 `np.sum`，反向把输出梯度 broadcast 回输入形状。
- `linear(x, W, b)`：先 `matmul(x, W)`，再可选加 bias。

这里的 `Linear` 权重形状与 PyTorch 略有不同：

- PyTorch 的 `nn.Linear(in, out)` 通常保存 `weight.shape == (out, in)`，前向使用 `x @ weight.T + bias`。
- `torch_1k.nn.Linear` 保存 `weight.shape == (in, out)`，前向直接使用 `x @ weight + bias`。

这种设计少写一次转置，更适合教学实现，但不是完全等价的 PyTorch 参数布局。

## 12. 索引和切片的梯度

切片由 `GetItem` 实现：

```python
y = x[index_or_slices]
```

前向直接执行 NumPy 索引。反向时，`gy` 只覆盖被切出来的区域，需要放回原始输入形状。源码通过 `Pad` 完成：

1. 创建一个原始输入形状的全零数组。
2. 把 `gy` 写回 `index_or_slices` 指定的位置。
3. 返回这个数组作为 `x` 的梯度。

因此：

```python
x = Tensor(np.arange(5))
y = x[1:3].sum()
y.backward()
```

会得到类似 `[0, 1, 1, 0, 0]` 的梯度。这个设计适合普通切片；如果未来要支持高级索引和重复索引，需要把“赋值回填”升级为“累加回填”。

## 13. no_grad、train、eval

`settings.py` 中的 `Config` 保存全局运行状态：

- `Config.enable_backprop`：是否在前向时记录计算图。
- `Config.train`：是否处于训练模式。

`torch_1k.no_grad()` 是 `using_config('enable_backprop', False)` 的上下文管理器。它和 PyTorch 的语义相似：上下文内的前向计算不会建立反向图。

`Module.train()` 会设置：

```python
Config.train = True
```

`Module.eval()` 会设置：

```python
Config.train = False
```

这与 PyTorch 当前语义保持一致：`model.eval()` 只影响 dropout、batch norm 等训练/推理行为，不会自动关闭 autograd；关闭 autograd 需要显式使用 `torch.no_grad()`。

## 14. nn 模块系统

`nn.Module` 提供了最小可用的模型封装：

- `__setattr__`：当赋值对象是 `Parameter` 或子 `Module` 时，把属性名记录到 `_parameters`。
- `parameters()`：遍历 `_parameters`，如果遇到子模块就递归 yield 它的参数。
- `__call__`：调用 `forward`，并用弱引用保存输入输出。
- `train()` / `eval()`：切换全局配置。

`Parameter` 只是 `Tensor` 的子类，不增加额外字段。它的作用是让 `Module.__setattr__` 区分“普通 Tensor”和“需要优化的参数”。

`Linear` 是 `Module` 的典型用法：

```python
self.weight = Parameter(...)
self.bias = Parameter(...)

def forward(self, x):
    return linear(x, self.weight, self.bias)
```

这样用户定义模型时，只要把子层挂到 `self.xxx` 上，`model.parameters()` 就能递归找到训练参数。

## 15. 损失函数

`MSELoss` 继承 `Function`，前向计算：

```python
mean((input - target) ** 2)
```

反向对输入返回：

```python
2 / input.size * (input - target) * grad_output
```

当前实现只返回 input 的梯度，不返回 target 的梯度。这符合常见监督学习中 target 不参与训练的使用方式，但如果要严格模拟 PyTorch 的可微 target 行为，需要返回 `(grad_input, grad_target)`。

## 16. 优化器

优化器分两层：

- `Optimizer`：保存参数列表，实现 `zero_grad()` 和 `step()` 模板。
- `SGD`：实现单个参数的更新规则。

`Optimizer.step()` 的逻辑是：

1. 遍历所有参数。
2. 跳过 `grad is None` 的参数。
3. 调用子类 `update_one(parameter)`。

`SGD.update_one` 使用最基础的梯度下降：

```python
parameter.data = parameter.data - lr * parameter.grad.data
```

`zero_grad()` 当前不是把梯度设为 `None`，而是把已有梯度数组填零。这两种方式都能避免梯度跨 step 累积，但语义与 PyTorch 默认行为略有差异。

源码中有 `MomentumSGD` 文件，但构造函数参数名和内部变量不一致，且没有从 `optim/__init__.py` 导出；当前可视为未完成实现。README 中提到 Adam，但当前源码没有 Adam 实现。

## 17. 数据加载器雏形

`utils/data` 中有两个类：

- `Dataset`：定义 `__getitem__` 和 `__len__` 抽象接口。
- `DataLoader`：按 batch size 取样本，支持 `shuffle` 和 `drop_last`。

`DataLoader.__next__` 会返回：

```python
features, labels
```

早期实现中二者都是从 batch 样本拆出来的 Python list。PLAN-006 后，`DataLoader` 支持标准迭代协议、`__len__` 和默认 batch collation；当数据集样本由 `Tensor` 组成时，batch 会自动通过 `torch.stack` 合并为 batched `Tensor`。典型用法为：

```python
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    ...
```

`TensorDataset.__getitem__` 与 PyTorch 一样返回 tuple，因此单输入数据集也会返回单元素 tuple。

## 18. 一个完整训练闭环

当前源码已经能表达最小线性回归训练：

```python
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

内部发生的事情是：

1. `model(x)` 调用 `Linear.forward`。
2. `Linear.forward` 调用 `functional.linear`。
3. `linear` 产生 `MatMul` 节点，bias 加法产生 `Add` 节点。
4. `MSELoss` 产生损失节点。
5. `loss.backward()` 沿 `MSELoss -> Add -> MatMul -> Parameter` 反传。
6. `SGD.step()` 读取 `Parameter.grad.data` 更新权重和偏置。

这就是 PyTorch 训练循环中最核心的“前向、损失、清梯度、反传、更新”五步。

## 19. 与 PyTorch 核心概念的对应关系

| PyTorch 概念 | torch_1k 对应实现 | 当前覆盖程度 |
| --- | --- | --- |
| `torch.Tensor` | `torch_1k.tensor.Tensor` | 支持 NumPy 数据、梯度、部分运算符 |
| `autograd.Function` | `torch_1k.function.Function` | 支持前向、反向、动态图连接 |
| 动态计算图 | `Tensor.creator` + `Function.inputs/outputs` | 支持基础动态图 |
| `backward()` | `Tensor.backward()` | 支持梯度累加、拓扑顺序、高阶导数 |
| `torch.no_grad()` | `torch_1k.no_grad()` | 支持关闭建图 |
| `nn.Module` | `torch_1k.nn.Module` | 支持参数收集和嵌套模块 |
| `nn.Parameter` | `torch_1k.nn.Parameter` | 继承 Tensor，默认 `requires_grad=True` |
| `nn.Linear` | `torch_1k.nn.Linear` | 支持线性层，权重布局不同 |
| `nn.MSELoss` | `torch_1k.nn.MSELoss` | 支持 input 梯度 |
| `optim.SGD` | `torch_1k.optim.SGD` | 支持基础 SGD |
| `DataLoader` | `torch_1k.utils.data.DataLoader` | 支持迭代、batch、shuffle、drop_last 和默认 Tensor collation |

## 20. 关键设计取舍

### 20.1 用 NumPy 做数值后端

优点是实现短、依赖少、容易阅读；缺点是不支持 GPU、设备管理、复杂 dtype 策略和真正的张量内核调度。

### 20.2 用 Python 对象保存动态图

每次前向执行都即时创建 `Function` 对象和输出 `Tensor`，因此实现直观，也支持普通 Python 控制流。代价是每个算子都有 Python 调度开销。

### 20.3 反向公式写在算子类中

每个算子的数学含义都集中在自己的 `backward` 方法中，适合教学和逐步扩展。代价是所有算子都要手写梯度公式，覆盖面取决于实现数量。

### 20.4 用弱引用和清梯度控制内存

弱引用可以减少循环引用，清理中间梯度可以降低内存占用。这让小框架也能跑一些重复训练循环。

### 20.5 用操作符重载模拟 PyTorch 体验

用户可以写 `x + y`、`x * y`、`x ** 2`，不用手动调用 `Add()`、`Mul()`。这提高了易用性，但也要求每个 Python 运算符都要明确注册。

## 21. 当前实现边界和风险

以下是当前实现仍然存在的边界，不是设计目标：

1. `DataLoader` 已支持 `__iter__`、`__len__` 和默认 Tensor collation；更完整的 sampler、多进程加载和 pinned memory 尚未实现。
2. `MSELoss` 当前只返回 input 梯度，不返回 target 梯度；教学训练主路径通常不需要 target 梯度。
3. 优化器已支持现有单参数组的 `state_dict()` / `load_state_dict()`，但还没有 PyTorch 完整的多 param group 体系。
4. `nn` 层还缺少 Dropout、BatchNorm、更多初始化工具和更完整的容器模块。
5. 规约与索引 API 仍只是常用子集，例如 `max`、`argmax` 的梯度语义和更多复杂索引尚未覆盖。
6. dtype、device 和 CUDA 行为只覆盖当前示例与测试所需的核心路径，尚未达到 PyTorch 完整语义。

这些边界不影响本项目作为“千行级 PyTorch 核心机制教学实现”的价值，但如果目标升级到更高兼容性，应优先修复。

## 22. 建议的后续演进顺序

如果继续推进到更接近 README 中的目标，建议按以下顺序做：

1. 增加 Dropout、BatchNorm 和初始化工具，让 `train()` / `eval()` 覆盖更多真实模块语义。
2. 继续扩展优化器到多 param group，并补齐更接近 PyTorch 的参数组超参数管理。
3. 继续扩展 `DataLoader` 到 sampler、自定义 batch sampler 和更完整的 PyTorch 参数兼容。
4. 补齐 `max`、`log_softmax`、更多 Tensor 方法和复杂索引。
5. 明确 `MSELoss` 对 target 是否需要梯度；若追求 PyTorch 语义，返回 target 梯度。

## 23. 总结

`torch_1k` 的核心价值不在于覆盖 PyTorch 的完整 API，而在于用极少代码展示深度学习框架最本质的机制：

- `Tensor` 保存数据、梯度和产生它的算子。
- `Function` 把前向计算、输出包装和计算图连接统一起来。
- `backward()` 从输出节点出发，按拓扑顺序反向调用各算子的梯度公式。
- 操作符重载、`Module`、`Parameter` 和 `SGD` 把自动微分核心包装成接近 PyTorch 的训练体验。

理解这套机制后，再扩展新算子、新网络层或新优化器，本质上就是沿着同一协议补充 `forward`、`backward`、参数收集和参数更新规则。
