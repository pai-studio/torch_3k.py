# torch 新手教程驱动 API 依赖设计

日期：2026-06-26

源码基准 Git 记录：`main@6b1c9bb3c61fb9657a4f3a6c1546e24eee79e2fc`

## 目标

本设计从“给 torch 新手写一组循序渐进的入门例子”倒推 `torch_1k` 最值得补齐的函数和类。目标不是追 PyTorch 全量 API，而是让教程中的核心概念都能用 PyTorch 风格代码表达，并且只替换 import 即可在 PyTorch 与 `torch_1k` 间切换。

## 新手教程场景

1. 张量基础：
   - 创建张量、查看 shape / dtype / device。
   - 使用 `arange`、`linspace`、`eye`、`zeros_like`、`randn_like`。
   - 使用 `reshape`、`T`、`matmul`、`sum(dim=..., keepdim=True)`、`mean(...)`。
   - 从 NumPy 数据进入 torch，再转回 NumPy / list。
2. 自动微分：
   - 用 `requires_grad=True` 创建输入。
   - 通过逐元素运算、规约、矩阵乘法构建标量 loss。
   - 调用 `backward()` 并查看 `.grad`。
3. 手写线性回归：
   - 用底层 Tensor 和矩阵乘法定义预测。
   - 手动清梯度、反传和更新。
   - 展示自动微分与优化器之前的底层机制。
4. `nn.Module` MLP 分类：
   - 使用 `nn.Sequential`、`nn.Linear`、`nn.Tanh` / `nn.Sigmoid` / `nn.ReLU`。
   - 使用 `nn.CrossEntropyLoss` 与 `optim.SGD`。
5. DataLoader 小批量训练：
   - 使用 `TensorDataset`、`DataLoader`、mini-batch loop。
6. CNN / Transformer 方向感：
   - 不要求每个教程都训练大模型，但要能解释最终模型依赖：CNN 依赖 `Conv2d`、`BatchNorm2d`、`MaxPool2d`、`Flatten`；序列模型依赖 `Embedding`、`LayerNorm`、`MultiheadAttention` / `TransformerEncoderLayer`。

## 依赖分层

### 底层必须稳固

这些能力直接影响计算图、梯度或设备一致性：

1. Tensor 容器：`data`、`grad`、`requires_grad`、`shape`、`dtype`、`device`。
2. 创建函数：`tensor`、`as_tensor`、`from_numpy`、`zeros`、`ones`、`randn`、`arange`、`linspace`、`eye`、`*_like`。
3. 算子：逐元素加减乘除、`matmul`、`sum`、`mean`、`reshape`、`transpose`、切片。
4. 自动微分：`Function.__call__` 建图、`Tensor.backward()` 拓扑反传、广播梯度还原。
5. `Module` / `Parameter` 参数注册与递归遍历。
6. `Optimizer.zero_grad()` / `step()`。

### 重要 wrapper

这些不改变底层数学能力，但决定真实 PyTorch 教程代码能否自然运行：

1. `Tensor.clone()` / `tolist()` / `matmul()` / `mm()`。
2. `torch.sum(input, dim=..., keepdim=...)` / `torch.mean(...)`。
3. `torch.randn_like()` / `torch.rand_like()`。
4. `nn.Tanh` / `nn.Sigmoid`。
5. `torch.as_tensor()` / `torch.from_numpy()`。
6. `torch.eye()`。

### 暂不追求

1. 完整 dtype promotion。
2. 原地操作族，如 `add_()`、`copy_()`。
3. 完整高级索引。
4. 完整 `torchvision` 数据集和 transform 体系。
5. 大规模模型训练性能。

## PLAN-038 选择

PLAN-036 / PLAN-037 已让完整训练脚本与入口层可运行。PLAN-038 应把入门教程中仍缺的基础 wrapper 和少量底层 Function 补齐，并新增一个教程式例子，作为“新手从张量到模型”的端到端验收。
