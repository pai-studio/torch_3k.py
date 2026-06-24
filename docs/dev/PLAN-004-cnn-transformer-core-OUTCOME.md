# PLAN-004 CNN/MNIST-like 与 Transformer 核心模块结果

日期：2026-06-24

## 完成内容

1. 优化器：
   - 修复并导出 `MomentumSGD`。
   - 新增 `Adam`。
   - 新增 `AdamW`。
2. CNN/MNIST：
   - 新增 `nn.Conv2d`。
   - 新增 `nn.MaxPool2d`。
   - 新增 `nn.Flatten`。
   - 新增 `examples/example5_mnist_cnn_train_compare.py`。
   - 默认使用本地 MNIST-like 数据，`USE_REAL_MNIST=1` 时下载并训练真实 MNIST 小子集。
3. Transformer：
   - 新增 `nn.Embedding`。
   - 新增 `nn.LayerNorm`。
   - 新增 `nn.MultiheadAttention`。
   - 新增 `nn.TransformerEncoderLayer`。
   - 新增 `examples/example6_transformer_train_compare.py`。
4. 基础算子：
   - `softmax`
   - `Tensor.transpose(dim0, dim1)`
   - `Tensor.permute`
   - `Tensor.__matmul__`
   - batch `matmul`
   - `sum(axis=...)` / `mean(axis=...)` 的基础反向
   - 反向传播跳过不可微输入的 `None` 梯度
5. 测试：
   - 新增 `tests/test_13_cnn_transformer.py`，覆盖常见 optimizer、CNN 反传、Transformer 反传。
6. 设计文档：
   - 更新 `docs/design/torch-3k-20260624-core-cuda-roadmap.md`，记录 PLAN-004 模块状态。

## 验证结果

已运行：

```bash
python examples/example5_mnist_cnn_train_compare.py
USE_TORCH_1K=0 python examples/example5_mnist_cnn_train_compare.py
USE_REAL_MNIST=1 python examples/example5_mnist_cnn_train_compare.py
USE_TORCH_1K=0 USE_REAL_MNIST=1 python examples/example5_mnist_cnn_train_compare.py
python examples/example6_transformer_train_compare.py
USE_TORCH_1K=0 python examples/example6_transformer_train_compare.py
pytest -q tests/test_13_cnn_transformer.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- `torch_1k` CNN 示例：`accuracy=1.000000`
- PyTorch CNN 示例：`accuracy=1.000000`
- `torch_1k` 真实 MNIST 小子集示例：`accuracy=1.000000`
- PyTorch 真实 MNIST 小子集示例：`accuracy=1.000000`
- `torch_1k` Transformer 示例：`accuracy=1.000000`
- PyTorch Transformer 示例：`accuracy=1.000000`
- 新增测试：`3 passed`
- 全量测试：`42 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## 说明

MNIST 示例默认使用本地生成的 28x28 MNIST-like 数据，避免下载依赖影响可重复运行；真实 MNIST 小子集路径已通过 `USE_REAL_MNIST=1` 验证。下载文件缓存于 `.gitignore` 已忽略的 `downloads/mnist/`。

## 代码规模

当前 `torch_1k` 核心代码统计约 `1779` 行，仍低于新定位建议的 3000 行以内。
