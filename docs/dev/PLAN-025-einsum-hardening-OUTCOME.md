# PLAN-025 einsum 经典用法与广播反向加固结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-025-einsum-hardening.md`

Git 基线：`a5d4d3b`

说明：本轮在包含 PLAN-021 至 PLAN-024 未提交改动的工作区状态上继续实施。

## 修复结果

1. 修复 `Einsum.backward` 对广播维度的处理：
   - 原问题：反向 `einsum` 得到的梯度形状如果大于输入原形状，代码直接 `reshape`，导致 `"ij,j->ij"` 这类广播场景报错。
   - 新逻辑：先按输入 spec 对齐 label 维度，再对前向广播出来的维度求和，最后对私有规约维 broadcast 回输入形状。
2. 新增 `_restore_einsum_grad_shape(...)`：
   - 统一处理反向梯度形状恢复。
   - 明确区分“广播维求和”和“缺失私有维 broadcast”。
3. 保持 PLAN-023 的支持边界不变：
   - 不支持 ellipsis。
   - 不支持同一输入内部 repeated labels / diagonal。
   - 不支持 PyTorch sublist equation 格式。

## 经典用法对比

新增 PyTorch 对比测试覆盖以下常用公式：

1. 矩阵乘：`"ij,jk->ik"`。
2. 批量矩阵乘：`"bij,bjk->bik"`。
3. 点积：`"i,i->"`。
4. 外积：`"i,j->ij"`。
5. 行规约：`"ij->i"`。
6. 全局规约：`"ij->"`。
7. 逐元素乘加：`"ij,ij->"`。
8. attention score：`"bqd,bkd->bqk"`。
9. attention context：`"bqk,bkd->bqd"`。
10. 广播门控：`"ij,j->ij"`。
11. 广播乘后规约：`"ij,j->i"`。
12. 广播维与私有规约维组合：`"ijk,j->i"`。

每个用法都对比：

- torch_1k 前向输出 vs PyTorch 前向输出。
- torch_1k 每个输入梯度 vs PyTorch 每个输入梯度。

## 新增示例

新增 `examples/example25_einsum_usage_compare.py`：

1. 同时导入 `torch_1k` 和 PyTorch。
2. 对每个经典用法输出：
   - 中文名称。
   - equation。
   - 输出 shape。
   - 前向最大误差。
   - 梯度最大误差。
3. 覆盖一个组合 attention pipeline：
   - `scores = einsum("bqd,bkd->bqk", query, key)`
   - `context = einsum("bqk,bkd->bqd", scores, value)`

## 验证结果

已运行：

```bash
python examples/example25_einsum_usage_compare.py
pytest -q tests/test_32_einsum_api.py
pytest -q tests/test_30_loss_functional_reduction.py tests/test_31_shape_split_chunk_repeat.py tests/test_32_einsum_api.py tests/test_33_topk_api.py
python -m compileall -q torch_1k examples
pytest -q
```

结果：

- 详细 einsum 示例：所有经典用法前向与梯度误差均在浮点舍入范围内。
- einsum 专项测试：`19 passed`
- PLAN-021 至 PLAN-025 相关测试集合：`38 passed`
- 全量测试：`168 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 ellipsis，例如 `"...ij,jk->...ik"`。
2. 尚未实现同一输入内部 repeated labels 的 diagonal / trace 语义，例如 `"ii->i"`。
3. 尚未实现 PyTorch sublist equation 格式。
4. 尚未实现完整 dtype promotion。

## 复核结论

`einsum` 是本库当前最重要的高阶张量算子之一。本轮修复了广播反向的关键缺陷，并用 PyTorch 经典用法进行前向与梯度逐项对比。当前核心子集已经能稳定覆盖矩阵乘、batch matmul、attention、点积、外积、规约和广播门控等真实训练场景。
