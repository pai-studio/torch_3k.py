# PLAN-031 nonzero 与 where(condition) 坐标 API 结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-031-nonzero-where-condition.md`

Git 基线：`81ffeb1`

## 实施结果

1. 新增 `torch_1k.nonzero(input, as_tuple=False)`：
   - 默认返回二维坐标 Tensor，shape 为 `(z, input.ndim)`。
   - `as_tuple=True` 返回每个维度一个 1D index Tensor。
   - 返回 Tensor 均为 `requires_grad=False`。
2. 扩展 `torch_1k.where(...)`：
   - 单参数 `where(condition)` 返回 `nonzero(condition, as_tuple=True)`。
   - 三参数 `where(condition, input, other)` 保留原有可微分流语义。
   - 只传两个参数时明确报错。
3. 新增 `Tensor.nonzero(as_tuple=False)` 方法。
4. 支持 bool mask 和 numeric input 的非零坐标。
5. 支持标量输入边界，输出 shape 与 PyTorch 对齐。
6. CPU/CUDA 共用同一套实现。

## 新增测试

新增 `tests/test_38_nonzero_where_condition.py`，覆盖：

1. bool mask 的 `nonzero(as_tuple=False)` 与 PyTorch 对比。
2. bool mask 的 `nonzero(as_tuple=True)` 与 PyTorch 对比。
3. numeric input 和 `Tensor.nonzero()`。
4. `where(condition)` 返回 tuple index。
5. 标量 `nonzero` shape。
6. 三参数 `where` 旧路径仍可反向传播。
7. 非 bool `where(condition)` 和部分参数报错。
8. CUDA 可选路径下坐标 Tensor 保持 CUDA 设备。

## 新增示例

新增 `examples/example31_nonzero_where_compare.py`：

1. 从正值 mask 中取二维坐标。
2. 用 `where(condition)` 得到 tuple index。
3. 用 numeric tensor 的 `nonzero()` 取非零坐标。
4. 同进程对比 `torch_1k` 与 PyTorch。

## 验证结果

已运行：

```bash
pytest -q tests/test_38_nonzero_where_condition.py
pytest -q tests/test_35_mask_indexing_numeric_api.py tests/test_37_scatter_api.py tests/test_38_nonzero_where_condition.py
pytest -q tests/test_35_mask_indexing_numeric_api.py tests/test_37_scatter_api.py tests/test_38_nonzero_where_condition.py tests/test_32_einsum_api.py tests/test_36_cross_entropy_advanced.py
python examples/example31_nonzero_where_compare.py
python examples/example30_scatter_compare.py
python examples/example29_einsum_ellipsis_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 nonzero / where(condition) 测试：`7 passed`
- mask/index/scatter/nonzero 相关测试：`21 passed`
- mask/index/scatter/nonzero、einsum 和高级交叉熵相关测试：`54 passed`
- 新增示例：通过，坐标和 tuple index 均与 PyTorch 一致
- 相邻功能示例：scatter 与 einsum ellipsis 示例均通过
- 全量测试：`210 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现完整高级索引语义。
2. 尚未实现 `out=` 参数。
3. 尚未实现完整 dtype promotion。

## 复核结论

本轮补齐了 mask/index 胶水 API 中缺失的坐标提取入口。现在比较 mask 可以通过 `nonzero` 或 `where(condition)` 转为索引坐标，与已有 `gather`、`scatter`、`masked_fill` 和三参数 `where` 组合使用，覆盖更多真实 PyTorch 代码中的条件索引写法。
