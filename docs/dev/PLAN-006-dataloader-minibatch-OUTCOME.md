# PLAN-006 数据加载与 mini-batch 训练结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-006-dataloader-minibatch.md`

Git 基线：`80cf9e44bd65861b0f83b35ee9f7a54f7ae748e4`

## 实施结果

1. 新增 `torch.stack(tensors, dim=0)`：
   - 支持正负维度范围检查。
   - 检查输入 Tensor 形状一致。
   - 反向传播会把梯度切回每个输入 Tensor。
2. 新增 `torch_1k.utils.data.TensorDataset`：
   - 支持一个或多个 Tensor/数组。
   - `__getitem__` 保持 PyTorch 的 tuple 返回语义。
3. 改造 `torch_1k.utils.data.DataLoader`：
   - 默认 batch collation 会把 Tensor、NumPy 数组和数字合并为 batched `Tensor`。
   - 支持 tuple/list/dict 样本结构的递归 collation。
   - 支持 `__len__`、每轮 `__iter__` 重置、`shuffle` 和 `drop_last`。
   - 保留 `collate_fn` 扩展点。
4. 补齐 `torch.utils.data` 访问形式：
   - `import torch_1k as torch` 后可以使用 `torch.utils.data.TensorDataset` 和 `torch.utils.data.DataLoader`。
5. 新增 mini-batch CNN 示例：
   - `examples/example7_mnist_dataloader_train_compare.py`
   - 使用同一训练循环跑通 `torch_1k` 与 PyTorch。
   - 默认使用本地 MNIST-like 数据，也支持 `USE_REAL_MNIST=1` 的真实 MNIST 小子集。
6. 更新设计文档中 data 层状态，记录 PLAN-006 已补齐默认 Tensor collation。

## 新增测试

新增 `tests/test_15_data_pipeline.py`，覆盖：

1. `torch.stack` 前向形状和反向梯度。
2. `TensorDataset` + `DataLoader` 自动返回 batched `Tensor`。
3. `drop_last` 和重复迭代行为。
4. NumPy 数组输入的 TensorDataset/DataLoader 路径。
5. `torch.utils.data` PyTorch 风格命名空间。

## 验证结果

已运行：

```bash
pytest -q tests/test_15_data_pipeline.py
python examples/example7_mnist_dataloader_train_compare.py
USE_TORCH_1K=0 python examples/example7_mnist_dataloader_train_compare.py
USE_REAL_MNIST=1 python examples/example7_mnist_dataloader_train_compare.py
USE_TORCH_1K=0 USE_REAL_MNIST=1 python examples/example7_mnist_dataloader_train_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增数据管线测试：`5 passed`
- mini-batch MNIST-like 示例：`torch_1k` 与 PyTorch 均 `accuracy=1.000000`
- mini-batch 真实 MNIST 小子集示例：`torch_1k` 与 PyTorch 均 `accuracy=1.000000`
- 全量测试：`51 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## 复核结论

本轮复核发现并修复了一个兼容性问题：NumPy 数组的 `.size` 是属性而不是方法，`TensorDataset` 现在只在 `.size` 可调用时按 Tensor 风格读取长度，否则回退到 `.shape[0]`。当前仍未覆盖 PyTorch DataLoader 的 sampler、多进程加载、pinned memory 等完整参数，这些属于后续兼容性扩展范围。
