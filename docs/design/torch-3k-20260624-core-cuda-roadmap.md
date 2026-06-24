# torch_3k 新定位与 CUDA 后端设计

日期：2026-06-24

源码基准 Git 记录：`main@701d4072f33d2fbdafbcac8bab49ddf3ae87307d`

## 1. 新定位

本库的新定位是：用尽量简洁、可读、教学友好的代码实现 PyTorch 核心功能。代码行数不再强制限制在 1000 行以内，但原则上整体保持在 3000 行以内较合适。

这里的“精简”应该体现在功能范围上，而不是体现在稳定性上：

- 减少非核心功能：不追求完整 PyTorch API、分布式、JIT、复杂 dtype 策略、图编译、混合精度等。
- 保留核心质量：核心训练链路应行为稳定、梯度正确、CPU/CUDA 后端一致、错误边界清晰。
- 面向真实场景：以“能训练一个小模型”为基准，而不是只演示标量求导。

## 2. 真实训练场景中的核心链路

一个真实但最小的深度学习训练任务通常包含：

1. 数据加载：从 `Dataset` / `DataLoader` 取 batch。
2. 张量创建：输入、标签、参数、临时中间量。
3. 模型前向：线性层、激活函数、矩阵乘法、广播、规约。
4. 损失计算：MSE 或交叉熵。
5. 反向传播：动态图拓扑遍历、梯度累加、广播梯度还原。
6. 参数更新：`zero_grad()`、`step()`、学习率。
7. 设备迁移：CPU 与 CUDA 间移动数据和模型。
8. 推理：`eval()`、`no_grad()`、输出转 NumPy。

因此，核心模块应该围绕这条链路组织，而不是围绕 PyTorch 的全部 API 面积展开。

## 3. 核心模块分层

### 3.1 后端与设备层

职责：

- 支持 NumPy CPU 数组。
- 可选支持 CuPy CUDA 数组。
- 提供统一的数组创建、数组模块识别、CPU/GPU 转换、设备查询。

建议接口：

- `get_array_module(x)`：返回 `numpy` 或 `cupy`。
- `ensure_array(data, device=None)`：把输入转为 CPU 或 CUDA 数组。
- `to_device(data, device)`：在 CPU/CUDA 间移动。
- `as_numpy(data)`：把任意后端数组转回 NumPy。
- `cuda_available()`：判断 CuPy 和 CUDA 是否可用。

这是 CUDA 支持的基础。上层算子不应该直接写死 `np.xxx`，而应根据输入数组选择 `xp.xxx`。

### 3.2 Tensor 层

职责：

- 保存 `data`、`grad`、`creator`、`generation`、`name`。
- 暴露 `shape`、`ndim`、`dtype`、`device`。
- 支持 `to()`、`cpu()`、`cuda()`。
- 通过操作符重载连接到可微算子。

Tensor 层不应该承担复杂数学逻辑，核心价值是作为动态图节点和设备感知的数据容器。

### 3.3 Function 与 Autograd 层

职责：

- 统一前向执行、输出包装、动态图连接。
- `Tensor.backward()` 按 generation 逆序调度。
- 支持梯度累加、释放中间梯度、高阶导数。
- 正确处理广播、切片、矩阵乘法等常用梯度。

这是本库最核心的教学价值，应该保持短小但可靠。

### 3.4 基础算子层

首批核心算子：

- 创建：`zeros`、`ones`、`rand`、`randn`、`linspace`、`normal`。
- 一元：`neg`、`exp`、`sin`、`cos`、`tanh`、`sigmoid`、`relu`。
- 二元：`add`、`sub`、`mul`、`div`、`pow`。
- 形状：`reshape`、`transpose`、`unsqueeze`、`broadcast_to`、`sum_to`。
- 线代：`matmul`、`linear`。
- 规约：`sum`，后续补 `mean`、`max`。
- 索引：基础切片与梯度回填。

优先保证这些算子的 CPU/CUDA 行为一致，再扩展更复杂算子。

### 3.5 nn 层

首批核心模块：

- `Parameter`
- `Module`
- `Linear`
- `MSELoss`
- `ReLU` 或 `functional.relu`

后续核心模块：

- `Sequential`
- `CrossEntropyLoss`
- `Softmax` / `LogSoftmax`
- 简单初始化工具
- `state_dict()` / `load_state_dict()`

阶段更新：PLAN-003 已补齐 `Sequential`、`ReLU` 和 `CrossEntropyLoss`，并用 XOR MLP 分类示例验证 PyTorch 导入替换路径。`Softmax` / `LogSoftmax` 暂时仍未作为独立算子暴露，交叉熵内部使用稳定 softmax 公式实现。

### 3.6 optim 层

首批核心优化器：

- `SGD`

后续核心优化器：

- `MomentumSGD`
- `Adam`

对教学代码而言，SGD 展示参数更新机制，Adam 展示真实训练常用状态型优化器。

### 3.7 data 层

核心能力：

- `Dataset` 抽象。
- `DataLoader` 标准迭代协议。
- batch 采样、shuffle、drop_last。

后续可考虑自动堆叠 batch，但应保持实现清晰。

## 4. CUDA/CuPy 支持策略

推荐用 CuPy 支持 CUDA，而不是直接编写 CUDA kernel。原因：

1. CuPy API 与 NumPy 高度相似，适合当前代码结构。
2. 上层算子只需把 `np.xxx` 改为 `xp.xxx`。
3. 能保持教学代码短小，避免陷入 CUDA kernel 细节。
4. CPU 和 CUDA 可以共享同一套自动微分逻辑。

首批 CUDA 支持应做到：

- `Tensor([1, 2]).cuda()` 返回 CUDA Tensor。
- `Tensor(..., device='cuda')` 可直接创建 CUDA Tensor。
- `Tensor.device` 返回 `cpu` 或 `cuda`。
- `Tensor.cpu()` 把 CUDA Tensor 移回 CPU。
- `Tensor.numpy()` 对 CUDA Tensor 自动转回 NumPy。
- `Module.to('cuda')` 递归迁移参数。
- 基础算子在 CUDA Tensor 上执行时保持 CUDA 数组输出。

如果环境没有 CuPy 或 CUDA，不应影响 CPU 路径测试。

## 5. 首批实施范围

本阶段优先实现以下内容：

1. 先编写 PyTorch 兼容示例，把示例作为接口验收标准。
2. 新增后端抽象模块，统一 NumPy/CuPy。
3. 扩展 `Tensor` 支持 `device`、`to()`、`cpu()`、`cuda()`、`detach()`。
4. 改造基础算子使用输入数组对应的后端模块。
5. 修复明显稳定性问题：
   - `nn.functional.relu` 语法错误。
   - `Sigmoid.backward` 未定义变量。
   - `Module.zero_grad` 拼写错误。
   - `Tensor.randn` 返回全零。
   - `DataLoader` 缺少 `__iter__`。
   - 二元算子广播梯度还原不完整。
6. 增加 CPU 回归测试和可选 CUDA 测试。

示例要求：训练主体代码与 PyTorch 保持一致，只允许在顶部切换：

```python
import torch_1k as torch
import torch_1k.nn as nn
import torch_1k.optim as optim
```

或：

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

## 6. 暂不纳入首批范围

以下能力重要，但不应阻塞 CUDA 后端基础落地：

- 完整 `torch.nn` API。
- 自动混合精度。
- 分布式训练。
- JIT/编译图。
- GPU kernel 手写优化。
- 完整 dtype promotion。
- 完整高级索引语义。

## 7. 判断完成的标准

本阶段完成后应满足：

1. 现有 CPU 测试通过或未引入新的 CPU 回归。
2. 无 CuPy 环境时，CUDA 相关测试自动跳过。
3. 有 CuPy/CUDA 环境时，基础 Tensor 创建、算子、反传、`Module.to('cuda')` 可用。
4. 文档记录新定位、核心模块边界和后续演进路线。
5. 代码仍保持教学可读性，不引入复杂框架或过度抽象。
