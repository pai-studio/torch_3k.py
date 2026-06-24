# PLAN-012 max API 兼容

日期：2026-06-24

Git 基线：`df6f5ca93991914fff15e3abf66c3be9a11a2389`

## 目标

补齐 PyTorch 常见 `max` API，支持分类推理和普通张量规约代码中的 `x.max()`、`x.max(dim=...)`、`torch.max(x, dim=...)` 和 `torch.max(x, y)`。示例继续保持只通过导入名切换 `torch_1k` 与 PyTorch。

## 当前缺口

1. `torch_1k` 只有 `argmax`，没有 `torch.max`。
2. `Tensor` 缺少 `x.max()` 和 `x.max(dim=...)` 方法。
3. `x.max(dim=...)` 需要返回 PyTorch 风格的 `values` 和 `indices`。
4. 设计文档仍把 `max` 记录为后续规约 API。

## 实施步骤

1. 在 `functional.numeric` 中实现可反向传播的规约 `max`：
   - `axis=None` 返回单个最大值。
   - `dim/axis` 返回 `MaxResult(values, indices)`。
   - 支持 `keepdim/keepdims`。
   - dim 规约反向传播到 argmax 位置。
2. 实现 elementwise `torch.max(x, y)`，并处理广播梯度还原。
3. 给 `Tensor` 增加 `max(dim=None, keepdim=False)` 方法。
4. 新增 `examples/example12_max_api_compare.py`，覆盖推理预测和可微 max loss。
5. 新增测试覆盖：
   - 全局 max 前向和反向。
   - dim max 的 values/indices、keepdim 和反向。
   - `torch.max(x, y)` elementwise 路径。
   - 默认数据 Tensor 不需要梯度时不建图。
6. 更新设计文档中规约 API 状态。
7. 运行新增测试、双后端示例、全量测试和编译检查。

## 验收标准

1. `x.max()` 返回最大值 Tensor。
2. `x.max(dim=1)` / `torch.max(x, dim=1)` 返回可解包、带 `.values` / `.indices` 的结果。
3. `values.sum().backward()` 能把梯度传回最大值位置。
4. `torch.max(x, y)` 支持 elementwise max 和反向传播。
5. 新增示例在 `torch_1k` 与 PyTorch 下均通过。
6. 全量测试通过。

## TODO 记录

1. 本计划暂不实现 PyTorch `torch.amax`。
2. 多维 tuple 规约不纳入本计划，后续如需更完整规约 API 再单独实现。
3. `max(dim=...)` 在重复最大值时按 `argmax` 位置传递梯度；全局 `max()` 对重复最大值按 mask 平分梯度。
