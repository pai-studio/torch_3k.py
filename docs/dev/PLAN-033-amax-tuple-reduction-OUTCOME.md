# PLAN-033 amax 与多维 tuple 最大值规约结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-033-amax-tuple-reduction.md`

Git 基线：`81ffeb1`

## 实施结果

1. 新增顶层函数：
   - `torch_1k.amax(input, dim=None, keepdim=False)`
2. 新增 Tensor 方法：
   - `Tensor.amax(dim=None, keepdim=False)`
3. 支持：
   - 全局规约。
   - 单维规约。
   - 多维 tuple/list 规约。
   - 负维度。
   - `keepdim=True`。
4. `amax` 梯度与 PyTorch 对齐：
   - 重复最大值位置平均分配梯度。
   - 多维 tuple 规约时在所有最大值位置分配梯度。
5. 保持 `torch.max(input, dim=...)` 旧路径不变：
   - 仍返回 values/indices。
   - 仍按 argmax 单位置回传梯度。
6. CPU/CUDA 共用同一套实现。

## 新增测试

新增 `tests/test_39_amax_api.py`，覆盖：

1. 全局 `amax` 的重复最大值梯度平均分配。
2. `amax(dim=(...))` tuple 维度规约前向和梯度与 PyTorch 对比。
3. `keepdim=True`。
4. 负维度。
5. `Tensor.amax(...)` 方法入口。
6. `torch.max(dim=...)` values/indices 和 argmax 梯度语义不回归。
7. 非法重复维度、越界维度和非法 dim 类型报错。
8. CUDA 可选路径下输出和梯度保持 CUDA 设备。

## 新增示例

新增 `examples/example33_amax_tuple_compare.py`：

1. 用 `(N, C, H, W)` 张量做空间维 `(H, W)` 最大值规约。
2. 保留 `keepdim=True` 的广播友好形状。
3. 同进程对比 `torch_1k` 与 PyTorch 的输出和梯度。

## 验证结果

已运行：

```bash
pytest -q tests/test_39_amax_api.py
pytest -q tests/test_21_max_api.py tests/test_39_amax_api.py
pytest -q tests/test_21_max_api.py tests/test_39_amax_api.py tests/test_32_einsum_api.py tests/test_35_mask_indexing_numeric_api.py tests/test_37_scatter_api.py tests/test_38_nonzero_where_condition.py tests/test_36_cross_entropy_advanced.py
python examples/example33_amax_tuple_compare.py
python examples/example32_einsum_repeated_labels_compare.py
python examples/example31_nonzero_where_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 amax 测试：`7 passed`
- max/amax 相关测试：`12 passed`
- max/amax、einsum、mask/index/scatter/nonzero 和高级交叉熵相关测试：`73 passed`
- 新增示例：通过，输出和梯度均与 PyTorch 一致
- 相邻功能示例：einsum repeated-label 与 nonzero/where 示例均通过
- 全量测试：`224 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. `torch.amin` / `torch.aminmax` 已在 PLAN-034 补齐。
2. 尚未实现 `out=` 参数。
3. 尚未实现完整 dtype promotion。

## 复核结论

本轮补齐了 values-only 最大值规约路径。现在用户可以用 `torch_1k.amax(x, dim=(...))` 表达多维空间/序列最大值聚合，同时保留 `torch.max(dim=...)` 的 indices 返回语义，两类 PyTorch API 边界更清晰。
