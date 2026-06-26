# PLAN-035 min API 兼容结果

日期：2026-06-26

计划文件：`docs/dev/PLAN-035-min-api.md`

Git 基线：`c448c68`

## 实施结果

1. 新增顶层函数：
   - `torch_1k.min(input)`
   - `torch_1k.min(input, dim, keepdim=False)`
   - `torch_1k.min(input, other)`
   - `torch_1k.minimum(input, other)`
2. 新增 Tensor 方法：
   - `Tensor.min(dim=None, keepdim=False)`
3. 新增返回类型：
   - `MinResult(values, indices)`
4. 支持：
   - 全局最小值规约。
   - 单维最小值规约，返回 `values` / `indices`。
   - 负维度。
   - `keepdim=True`。
   - elementwise min。
   - 广播输入的 elementwise min 梯度还原。
5. 梯度语义与 PyTorch 对齐：
   - 全局 `min(input)` 对重复最小值平均分配梯度。
   - `min(input, dim=...)` 只向 argmin 位置回传梯度。
   - elementwise min 在相等位置对两个输入平均分配梯度。
6. CPU/CUDA 共用同一套实现。

## 新增测试

新增 `tests/test_41_min_api.py`，覆盖：

1. 全局 `min` 重复最小值梯度平均分配。
2. `min(dim=...)` values/indices 和 argmin 梯度。
3. `keepdim=True`。
4. 负维度。
5. `Tensor.min(...)` 方法入口。
6. elementwise `torch.min(input, other)` 与 `torch.minimum(input, other)`。
7. elementwise tie 梯度。
8. 广播梯度还原。
9. tuple dim 报错。
10. CUDA 可选路径。

## 新增示例

新增 `examples/example35_min_compare.py`：

1. 对二维 score 张量执行 `torch_1k.min(x, dim=1)`。
2. 对 score 与 cap 张量执行 `torch_1k.minimum(x, cap)`。
3. 同进程对比 `torch_1k` 与 PyTorch 的 values、indices、elementwise 输出和梯度。

## 验证结果

已运行：

```bash
pytest -q tests/test_41_min_api.py
pytest -q tests/test_21_max_api.py tests/test_39_amax_api.py tests/test_40_amin_aminmax_api.py tests/test_41_min_api.py
python examples/example35_min_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 min 测试：`8 passed`
- max/amax/amin/aminmax/min 相关测试：`30 passed`
- 新增示例：通过，输出和梯度均与 PyTorch 一致
- 全量测试：`242 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 `out=` 参数。
2. 尚未实现 named tensor dim。
3. 尚未实现完整 dtype promotion。

## 复核结论

本轮补齐了 `max` 对称的最小值 API。现在用户可以使用 PyTorch 风格的全局 `torch_1k.min(x)`、按维度 `torch_1k.min(x, dim=...)` 和 elementwise `torch_1k.minimum(x, y)`；分类、检索、数值范围检查和裁剪类代码里的常见最小值路径可以直接替换导入名运行。
