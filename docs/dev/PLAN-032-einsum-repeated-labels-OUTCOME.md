# PLAN-032 einsum repeated labels / diagonal 常用路径结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-032-einsum-repeated-labels.md`

Git 基线：`81ffeb1`

## 实施结果

1. `torch.einsum` 支持同一输入内部 repeated labels：
   - `"ii->i"`：提取 diagonal。
   - `"ii->"`：trace / 对角求和。
   - `"ijj->i"`：沿重复标签取对角后规约。
   - `"ii,i->"`：多输入组合。
2. repeated labels 与 ellipsis 可组合：
   - `"...ii->...i"`
   - `"...ii->..."`
3. 前向使用 NumPy/CuPy `einsum` 的 repeated-label 语义。
4. 反向传播新增 unique-label 梯度路径：
   - 先对 repeated-label 输入构造唯一标签 spec。
   - 用既有反向 `einsum` 公式计算唯一标签空间梯度。
   - 再把梯度 scatter 回原输入 diagonal。
   - 非对角线位置梯度为 0。
5. repeated label 对应轴尺寸不一致时明确报错。
6. 保持已有非 repeated-label、broadcast 和 ellipsis 路径不回归。

## 新增测试

更新 `tests/test_32_einsum_api.py`，新增覆盖：

1. `"ii->i"` diagonal 前向与梯度。
2. `"ii->"` trace 前向与梯度。
3. `"ijj->i"` repeated label 规约。
4. `"ii,i->"` 多输入 repeated-label 组合。
5. `"...ii->...i"` ellipsis + repeated label。
6. `"...ii->..."` batch trace。
7. repeated-label 轴尺寸不一致报错。
8. CUDA 可选路径下 repeated-label 输出和梯度保持 CUDA 设备。

## 新增示例

新增 `examples/example32_einsum_repeated_labels_compare.py`：

1. 在同一进程中分别运行 `torch_1k` 与 PyTorch。
2. 覆盖 diagonal、trace、重复标签规约、多输入组合和 batch trace。
3. 对比输出和所有输入梯度。

## 验证结果

已运行：

```bash
pytest -q tests/test_32_einsum_api.py
pytest -q tests/test_32_einsum_api.py tests/test_35_mask_indexing_numeric_api.py tests/test_37_scatter_api.py tests/test_38_nonzero_where_condition.py tests/test_36_cross_entropy_advanced.py
python examples/example32_einsum_repeated_labels_compare.py
python examples/example29_einsum_ellipsis_compare.py
python examples/example31_nonzero_where_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- `einsum` 专项测试：`33 passed`
- einsum、mask/index/scatter/nonzero 和高级交叉熵相关测试：`61 passed`
- 新增 repeated-label 示例：通过，输出和梯度均与 PyTorch 一致
- ellipsis 示例：通过，确认 PLAN-029 路径无回归
- nonzero/where 示例：通过，确认相邻索引路径无回归
- 全量测试：`217 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 PyTorch sublist equation 格式。
2. 尚未实现 `out=` 参数。
3. 尚未实现完整 dtype promotion。

## 复核结论

本轮把 `einsum` 的核心语义进一步补全到 repeated-label diagonal / trace 路径。现在 `torch_1k.einsum` 已覆盖矩阵乘、batch matmul、attention、ellipsis 泛化 batch 维、广播反向，以及 diagonal / trace 类写法，仍保持实现结构清晰可读。
