# PLAN-026 sort / argsort 排序 API 结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-026-sort-argsort-api.md`

Git 基线：`a5d4d3b`

说明：本轮在包含 PLAN-021 至 PLAN-025 未提交改动的工作区状态上继续实施。

## 实施结果

1. 新增顶层 `torch_1k.sort(input, dim=-1, descending=False, stable=False)`。
2. 新增 `Tensor.sort(dim=-1, descending=False, stable=False)`。
3. 新增顶层 `torch_1k.argsort(input, dim=-1, descending=False, stable=False)`。
4. 新增 `Tensor.argsort(dim=-1, descending=False, stable=False)`。
5. `sort` 返回 PyTorch 风格 `SortResult(values, indices)`。
6. 支持默认最后一维、显式 `dim`、负维度和 `descending=True`。
7. 支持标量 Tensor：
   - `sort` 返回原值和索引 `0`。
   - `argsort` 返回索引 `0`。
8. values 支持反向传播：
   - 上游梯度通过 `put_along_axis` scatter 回输入排序前的位置。
9. indices / argsort 结果不追踪梯度。
10. 新增 `examples/example26_sort_argsort_compare.py`：
    - 同进程对比 `torch_1k` 与 PyTorch。
    - 输出 ascending values、descending values、indices、argsort ranking 和梯度最大误差。
11. 更新设计文档，记录 PLAN-026 已补齐排序 API。
12. 更新早期结果文档中过期的排序 TODO：
    - PLAN-013 结果文档记录 `topk`、`sort`、`argsort` 已在后续计划补齐。
    - PLAN-024 结果文档记录 `sort` / `argsort` 核心子集已由 PLAN-026 补齐。

## 新增测试

新增 `tests/test_34_sort_argsort_api.py`，覆盖：

1. 升序排序的 values、indices 和梯度与 PyTorch 一致。
2. 降序排序和负维度与 PyTorch 一致。
3. `Tensor.sort(...)` 方法。
4. `stable=True` 兼容签名。
5. 顶层 `torch.argsort(...)` 和 `Tensor.argsort(...)`。
6. 标量 Tensor 的 `sort` / `argsort`。
7. 非法维度错误。
8. 可选 CUDA 路径下 values、indices 和梯度保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_34_sort_argsort_api.py
python examples/example26_sort_argsort_compare.py
pytest -q tests/test_30_loss_functional_reduction.py tests/test_31_shape_split_chunk_repeat.py tests/test_32_einsum_api.py tests/test_33_topk_api.py tests/test_34_sort_argsort_api.py
python -m compileall -q torch_1k examples
pytest -q
```

结果：

- 新增 sort / argsort 测试：`7 passed`
- 新增 sort / argsort 示例：与 PyTorch 的 values、indices、argsort 和梯度误差均在浮点舍入范围内
- PLAN-021 至 PLAN-026 新增测试集合：`45 passed`
- 全量测试：`175 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. `stable=True` 当前只作为兼容签名接收，不承诺稳定排序语义。
2. tie 场景未保证与 PyTorch 在所有后端上完全一致；不同后端排序稳定性可能不同。
3. 尚未实现 `out=` 参数。
4. 尚未实现完整 dtype promotion。

## 复核结论

本轮补齐了排序族的核心子集。`sort` / `argsort` 现在可以覆盖 logits 排名、检索排序、序列调试和排序后梯度回传等常见场景，并和 `topk` 一起形成完整的预测索引教学路径：values 可微，indices 不可微，CPU/CUDA 后端共享同一套实现结构。
