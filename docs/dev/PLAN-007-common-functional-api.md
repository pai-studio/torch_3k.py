# PLAN-007 常用函数式 API 兼容

日期：2026-06-24

Git 基线：`d8c16cd3dfca4b28cbf504705946c8882271b67f`

## 目标

补齐高频 PyTorch 函数式 API，让常见代码中的 `torch.log`、`torch.relu`、`x.log()`、`x.relu()`、`nn.functional.relu(...)` 都能在 `torch_1k` 下直接运行。示例仍要求只通过顶部导入名切换 `torch_1k` 与 PyTorch。

## 当前缺口

1. README 已把 `log`、`relu` 等常用函数列为核心功能，但顶层 `torch_1k.log` 和 `torch_1k.relu` 尚不完整。
2. `Tensor` 缺少常用方法形式：`x.log()`、`x.relu()`、`x.exp()` 等。
3. `torch_1k.nn.functional.relu` 目前单独实现，和顶层函数式 API 存在重复。
4. 现有示例主要覆盖训练模块，缺少对函数式代码替换的独立验收。

## 实施步骤

1. 在 `torch_1k.functional.numeric` 中实现可反向传播的 `log` 和 `relu`。
2. 让 `torch_1k.nn.functional.relu` 复用统一实现。
3. 给 `Tensor` 增加 `exp()`、`log()`、`relu()`、`tanh()`、`sigmoid()` 方法。
4. 新增 `examples/example8_functional_api_compare.py`，覆盖顶层函数、Tensor 方法和 `nn.functional`。
5. 补充测试，验证 `log` 和 `relu` 的前向值与反向梯度。
6. 更新 README 与设计文档中的常用函数状态。
7. 运行新增示例双后端、全量测试和编译检查。

## 验收标准

1. `torch.log(x)`、`torch.relu(x)`、`x.log()`、`x.relu()` 均可运行。
2. `log` 的梯度为 `1 / x`。
3. `relu` 的负数与零点梯度为 `0`，正数梯度为 `1`。
4. 新增函数式示例在 `torch_1k` 和 PyTorch 下均通过。
5. 全量测试通过。
