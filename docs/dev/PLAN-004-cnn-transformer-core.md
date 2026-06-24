# PLAN-004 CNN/MNIST-like 与 Transformer 核心模块

日期：2026-06-24

## 目标

继续补齐 PyTorch 核心教学实现，增加常见 optimizer 和经典模块，使 CNN 分类示例和 Transformer 序列分类示例都能按 PyTorch 风格直接跑通。

## 范围

本阶段聚焦“能支撑真实训练闭环的最小核心子集”，不追求完整 PyTorch API。

### Optimizer

1. 修复并导出 `MomentumSGD`。
2. 新增 `Adam`。
3. 新增 `AdamW`。

### CNN/MNIST-like

1. 新增 `nn.Conv2d`，支持教学版 NCHW 输入、stride、padding。
2. 新增 `nn.MaxPool2d`，支持教学版 2D 最大池化。
3. 新增 `nn.Flatten`。
4. 新增 PyTorch 兼容 CNN 分类示例，默认使用 28x28 本地合成数字样式数据；设置 `USE_REAL_MNIST=1` 时下载并训练真实 MNIST 小子集。

### Transformer

1. 新增 `nn.Embedding`。
2. 新增 `nn.LayerNorm`。
3. 新增 `nn.MultiheadAttention`。
4. 新增 `nn.TransformerEncoderLayer`。
5. 新增 PyTorch 兼容 Transformer 序列分类示例。

### 基础算子

按示例需要补齐：

1. `softmax`
2. `transpose(dim0, dim1)` / `permute`
3. `matmul` 的 batch 输入支持
4. `mean(axis=...)` / `sum(axis=...)` 的基础反向
5. `Embedding` 所需的索引梯度回填

## 验收标准

1. `python examples/example5_mnist_cnn_train_compare.py` 跑通 `torch_1k` 路径。
2. `USE_TORCH_1K=0 python examples/example5_mnist_cnn_train_compare.py` 跑通 PyTorch 路径。
3. `python examples/example6_transformer_train_compare.py` 跑通 `torch_1k` 路径。
4. `USE_TORCH_1K=0 python examples/example6_transformer_train_compare.py` 跑通 PyTorch 路径。
5. 新增测试覆盖 optimizer、CNN 模块、Transformer 模块。
6. 全量测试通过。

## 说明

本阶段的 MNIST 示例默认使用本地合成的 28x28 数字样式数据，避免下载真实 MNIST 导致测试不稳定；同时提供 `USE_REAL_MNIST=1` 的真实 MNIST 小子集路径，下载文件缓存于 `downloads/mnist/`。
