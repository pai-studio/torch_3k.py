# PLAN-036 PyTorch 训练脚本替换兼容基线结果

日期：2026-06-26

计划文件：`docs/dev/PLAN-036-pytorch-training-compat-baseline.md`

实施基线 Git：`e7881de1d8e8cd555ecdbf10871cabb3d26f556e`

结果提交 Git：`baf1c01fa1d088787149bb7e1b68db0b1cc2e8b5`

## 实现内容

1. 新增 `examples/example36_pytorch_training_baseline.py`，作为训练脚本替换兼容基线：
   - 默认使用 `torch_1k`。
   - `USE_TORCH_1K=0` 时切换到 PyTorch。
   - 训练主体覆盖 MLP、CNN、Transformer 三条路径。
   - 覆盖 `TensorDataset` / `DataLoader` mini-batch、`Sequential`、`Linear`、`BatchNorm1d`、`BatchNorm2d`、`Dropout`、`Conv2d`、`MaxPool2d`、`Flatten`、`Embedding`、`TransformerEncoderLayer`、`CrossEntropyLoss`、`Adam`、`AdamW`。
   - 覆盖 `train()` / `eval()`、`no_grad()`、`argmax` 评估、`state_dict()`、`load_state_dict()`、`torch.save()`、`torch.load()`。
2. 新增 `tests/test_42_pytorch_training_baseline.py`：
   - 直接调用示例的 `torch_1k` 路径。
   - 使用 subprocess 运行 `USE_TORCH_1K=0` 的 PyTorch 路径。
3. 更新 README：
   - 将原先“只替换 torch 可做分类训练”的未完成项改为已完成的训练脚本替换兼容基线。
4. 更新设计路线图：
   - 记录 PLAN-036 已新增跨模块训练基线，覆盖数据加载、模型前向、损失、反传、优化、评估、checkpoint 和 PyTorch 导入替换路径。

## 验证结果

1. 默认 `torch_1k` 示例：
   - 命令：`python examples/example36_pytorch_training_baseline.py`
   - 结果：通过
   - MLP / CNN / Transformer accuracy 均为 `1.000000`
   - MLP checkpoint error 为 `0.00000000`
2. PyTorch 导入替换路径：
   - 命令：`USE_TORCH_1K=0 python examples/example36_pytorch_training_baseline.py`
   - 结果：通过
   - MLP / CNN / Transformer accuracy 均为 `1.000000`
   - MLP checkpoint error 为 `0.00000000`
3. 新增测试：
   - 命令：`pytest -q tests/test_42_pytorch_training_baseline.py`
   - 结果：`2 passed`
4. 全量测试：
   - 命令：`pytest -q`
   - 结果：`244 passed`
5. 编译检查：
   - 命令：`python -m compileall -q torch_1k examples`
   - 结果：通过

## TODO / 未完成事项

1. 仍不实现完整 `out=` 参数。
2. 仍不实现 named tensor dim。
3. 仍不实现完整 dtype promotion。
4. 仍不实现多进程 DataLoader、pinned memory 或 distributed sampler。
5. checkpoint 只验证同一后端内保存恢复，不保证 PyTorch 与 `torch_1k` 文件跨后端互相加载。

## 复核结论

本轮没有继续追加单个函数，而是把已有 API 收束为一条跨模块训练基线。现在同一份训练主体可以在 `torch_1k` 和 PyTorch 下分别运行 MLP、CNN、Transformer 分类训练，并完成评估和 checkpoint roundtrip。这个基线比继续补 `out=` 或 named dim 更贴近当前项目定位：用教学友好的代码支撑真实但最小的 PyTorch 训练工作流。
