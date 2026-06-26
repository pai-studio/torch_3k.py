# PLAN-029 einsum ellipsis 常用路径结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-029-einsum-ellipsis.md`

Git 基线：`81ffeb1`

## 实施结果

1. `torch.einsum` 新增 ellipsis 支持：
   - 每个输入 spec 最多一个 `...`。
   - 显式输出可包含 `...`。
   - 隐式输出会把 ellipsis 维保留在最前面。
2. 支持 ellipsis 维度广播：
   - 多输入 ellipsis 维按右对齐语义展开。
   - operand 缺失的 ellipsis 维在反向传播中通过既有 `einsum` backward 自动规约。
3. 保持现有普通标签路径不变：
   - 仍不支持同一输入内部 repeated labels / diagonal。
   - 仍不支持 PyTorch sublist equation 格式。
4. 内部实现把 ellipsis 展开成不与用户标签冲突的普通标签，再复用 NumPy/CuPy `einsum` 和现有反向传播恢复逻辑。
5. 保持 CPU/CUDA 共用同一套实现。

## 新增测试

更新 `tests/test_32_einsum_api.py`，新增覆盖：

1. `"...ij,jk->...ik"`：任意 batch 维矩阵乘。
2. `"...ij,...jk->...ik"`：两侧都有 ellipsis 的 batch matmul。
3. ellipsis broadcast：`(1, 3, 4)` 与 `(2, 4, 5)` 广播到 `(2, 3, 5)`。
4. `"...ij,jk"`：隐式输出中的 ellipsis。
5. `"...ij->..."`：保留 ellipsis 维并规约普通标签。
6. `"...i->i"`：显式输出不含 ellipsis 时规约 batch 维。
7. 非法 ellipsis 写法报错。
8. CUDA 可选路径下 ellipsis 输出和梯度保持 CUDA 设备。

## 新增示例

新增 `examples/example29_einsum_ellipsis_compare.py`：

1. 在同一进程中分别运行 `torch_1k` 与 PyTorch。
2. 覆盖泛化 batch matmul、ellipsis broadcast、attention score 和 ellipsis 规约。
3. 对比每个 case 的输出和所有输入梯度。

## 验证结果

已运行：

```bash
pytest -q tests/test_32_einsum_api.py
pytest -q tests/test_32_einsum_api.py tests/test_35_mask_indexing_numeric_api.py tests/test_36_cross_entropy_advanced.py
python examples/example29_einsum_ellipsis_compare.py
python examples/example25_einsum_usage_compare.py
python examples/example28_cross_entropy_advanced_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- `einsum` 专项测试：`26 passed`
- `einsum`、mask/numeric 和高级交叉熵相关测试：`41 passed`
- 新增 ellipsis 示例：通过，最大输出差 `3.552713678801e-15`，最大梯度差 `8.881784197001e-16`
- 经典 einsum 示例：通过，既有用法无回归
- 高级交叉熵示例：通过，确认上一轮改动仍正常
- 全量测试：`197 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 同一输入内部 repeated labels / diagonal 已在 PLAN-032 补齐，例如 `"ii->i"`。
2. 尚未实现 PyTorch sublist equation 格式。
3. 尚未实现完整 dtype promotion。
4. 尚未实现 `out=` 参数。

## 复核结论

本轮把 `einsum` 从固定 batch 标签扩展到 PyTorch 常用的泛化 ellipsis 写法。现在用户可以用 `"...ij,...jk->...ik"` 和 `"...qd,...kd->...qk"` 表达任意 batch 维矩阵乘与 attention score，同时保持已有自动微分实现可读、可验证。
