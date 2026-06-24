# PLAN-009 requires_grad 语义兼容结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-009-requires-grad.md`

Git 基线：`c239444653d32363c70bcd78ea3f86753d93e5d7`

## 实施结果

1. `Tensor` 新增 `requires_grad` 属性和 `requires_grad_()` 方法。
2. `torch.tensor`、`rand`、`randn`、`zeros`、`ones`、`linspace`、`normal` 等公开创建函数默认 `requires_grad=False`，并支持显式 `requires_grad=True`。
3. 保持 `Tensor(...)` 直接构造的内部默认可求导行为，避免破坏项目原有底层 autograd 测试。
4. `ensure_tensor` 创建的 Python/NumPy 常量默认不需要梯度，避免常量和普通输入数据无谓建图。
5. `nn.Parameter` 默认 `requires_grad=True`。
6. `Function.__call__` 现在只有在全局允许建图且至少一个输入需要梯度时才连接动态图。
7. `backward()` 只给 `requires_grad=True` 的输入累积梯度；`torch.no_grad()` 会阻止显式 requires_grad 输入建图。
8. `to()`、`detach()`、`float()`、`long()`、`argmax()`、比较、state dict 和 optimizer state 保存路径已按 PyTorch 语义传播或断开 `requires_grad`。
9. `torch.unsqueeze()` 对 Tensor 输入改为基于 `reshape`，显式需要梯度的输入不会丢计算图。
10. 新增 `examples/example9_requires_grad_compare.py`，同一代码可在 `torch_1k` 与 PyTorch 下运行。
11. 更新 README 和设计文档，记录 PLAN-009 后的 autograd 建图条件和剩余边界。

## 测试调整

部分旧测试原本依赖 `torch.tensor(...)` 或 NumPy 输入隐式建图。PLAN-009 后这些路径按 PyTorch 语义默认不需要梯度，因此已调整为：

1. 真正测试输入梯度的用例显式传入 `requires_grad=True`。
2. NumPy 输入的 `F.sum` 用例改为验证不建图；需要梯度的分支继续使用 `Tensor(...)`。

## 新增测试

新增 `tests/test_18_requires_grad.py`，覆盖：

1. `torch.tensor` 默认不追踪梯度。
2. `torch.tensor(..., requires_grad=True)` 可反向传播。
3. `nn.Parameter` 默认追踪梯度。
4. 模型输入数据默认不积累梯度，参数正常积累梯度。
5. `torch.no_grad()` 覆盖显式 requires_grad 输入。
6. `requires_grad_()`、`to()`、`detach()`、dtype 转换的传播或断开语义。
7. DataLoader batch 和 model state dict 默认是纯数据 Tensor。

## 验证结果

已运行：

```bash
pytest -q tests/test_18_requires_grad.py
pytest -q tests/test_11_pytorch_compat.py tests/test_12_mlp.py tests/test_15_data_pipeline.py tests/test_16_functional_api.py
python examples/example9_requires_grad_compare.py
USE_TORCH_1K=0 python examples/example9_requires_grad_compare.py
python examples/example4_mlp_train_compare.py
python examples/example7_mnist_dataloader_train_compare.py
python examples/example8_functional_api_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 requires_grad 测试：`6 passed`
- 受影响梯度测试：`13 passed`
- requires_grad 示例：`torch_1k` 与 PyTorch 均通过
- MLP / DataLoader CNN / 函数式示例均通过
- 全量测试：`63 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## 复核结论

本轮变更显著减少了普通数据 Tensor 的无谓建图，使公开 API 更接近 PyTorch。当前仍保留 `Tensor(...)` 直接构造默认可求导，主要是为了兼容本项目底层 autograd 教学测试；用户侧 PyTorch 风格代码应优先使用 `torch.tensor(...)`。后续高价值目标是增加 Dropout/BatchNorm 等训练态模块，让 `model.train()` / `model.eval()` 的差异具备真实行为承载。
