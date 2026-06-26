# PLAN-012 max API 兼容结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-012-max-api.md`

Git 基线：`df6f5ca93991914fff15e3abf66c3be9a11a2389`

## 实施结果

1. 新增可反向传播的全局规约 `torch.max(x)`：
   - 返回最大值 Tensor。
   - 反向传播会把梯度传回最大值位置；重复最大值时按 mask 平分梯度。
2. 新增 `Tensor.max(dim=..., keepdim=False)` 和 `torch.max(x, dim=..., keepdim=False)`：
   - 返回 `MaxResult(values, indices)`。
   - 支持 tuple 解包。
   - 支持 `.values` / `.indices` 属性。
   - `indices` 不追踪梯度。
   - `values` 的反向传播传到 `argmax` 位置。
3. 新增 elementwise `torch.max(x, y)`：
   - 支持广播。
   - 反向传播会按输入原始 shape 还原梯度。
4. 新增 `Tensor.max()` 方法。
5. 新增 `examples/example12_max_api_compare.py`：
   - 覆盖 `logits.max(dim=1)` 推理预测。
   - 覆盖 `torch.max(x, y)` elementwise 路径。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
6. 更新设计文档：
   - 将基础规约更新为 `sum`、`mean`、`max`。
   - 将剩余规约/索引 TODO 改为 tuple 规约、复杂索引和 `argmax` 梯度语义。

## 新增测试

新增 `tests/test_21_max_api.py`，覆盖：

1. 全局 `torch.max(x)` 前向和反向。
2. `x.max(dim=1)` 的 values、indices、tuple 解包和反向。
3. `torch.max(x, dim=1, keepdim=True)` 的 shape。
4. `torch.max(x, y)` elementwise 前向和反向。
5. 普通数据 Tensor 的 max 不建图。

## 验证结果

已运行：

```bash
pytest -q tests/test_21_max_api.py
python examples/example12_max_api_compare.py
USE_TORCH_1K=0 python examples/example12_max_api_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 max API 测试：`5 passed`
- max API 双后端示例：`torch_1k` 与 PyTorch 输出一致
- 全量测试：`75 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. `torch.amax` 已在 PLAN-033 补齐。
2. 多维 tuple 最大值规约已在 PLAN-033 通过 `amax(dim=(...))` 补齐。
3. `torch.max(x, dim=...)` 的重复最大值梯度使用 `argmax` 位置；全局 `torch.max(x)` 对重复最大值平分梯度。
4. 复杂索引和 `argmax` 梯度语义仍属于后续规约/索引 API 扩展范围。

## 复核结论

本轮补齐了分类推理和常见规约代码中的高频 `max` API。当前实现覆盖最常见的 PyTorch 替换路径：全局 max、按维度 max、keepdim、values/indices 返回和 elementwise max。
