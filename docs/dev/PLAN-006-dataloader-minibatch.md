# PLAN-006 数据加载与 mini-batch 训练

日期：2026-06-24

Git 基线：`80cf9e44bd65861b0f83b35ee9f7a54f7ae748e4`

## 目标

补齐 PyTorch 风格数据管线，让训练示例不再只能整批喂入 `Tensor`，而是可以用 `TensorDataset` 和 `DataLoader` 组织 mini-batch 训练。示例代码必须保持 PyTorch 兼容，只通过顶部导入名切换 `torch_1k` 与 PyTorch。

## 当前缺口

1. `torch_1k.utils.data` 缺少 `TensorDataset`。
2. `DataLoader` 当前返回 Python list，不能像 PyTorch 默认 collate 一样返回 batched `Tensor`。
3. `DataLoader` 缺少 `__len__`，不方便训练循环统计 batch 数。
4. 顶层缺少常用的 `torch.stack`，DataLoader 默认 collate 和用户代码都需要它。
5. MNIST/CNN 示例仍是整批训练，没有覆盖真实 mini-batch 数据流。

## 实施步骤

1. 新增可反向传播的 `torch.stack(tensors, dim=0)`。
2. 新增 `TensorDataset`，保持 `__getitem__` 返回 tuple 的 PyTorch 语义。
3. 改造 `DataLoader`：
   - 支持 `__iter__` 每轮重置。
   - 支持 `__len__`。
   - 默认 collate `Tensor`、NumPy 数组、数字、tuple/list 和 dict。
   - 保留 `shuffle` 和 `drop_last`。
4. 新增 `examples/example7_mnist_dataloader_train_compare.py`，使用 `TensorDataset` 和 `DataLoader` 做 mini-batch CNN 训练。
5. 补充测试覆盖 `stack` 反向传播、TensorDataset/DataLoader batch 形状、`drop_last` 与重复迭代。
6. 更新设计文档中的 data 层阶段状态。
7. 运行新增示例的 `torch_1k` 与 PyTorch 双路径、全量测试和编译检查。

## 验收标准

1. `torch.stack` 可在常用维度上拼接 Tensor，并能把梯度传回每个输入。
2. `for x, y in DataLoader(TensorDataset(...), batch_size=...)` 返回 batched `Tensor`。
3. 同一训练循环可通过替换导入名在 `torch_1k` 和 PyTorch 下运行。
4. mini-batch CNN 示例默认数据集准确率达到 `0.95` 以上。
5. 全量测试通过。
