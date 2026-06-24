# PLAN-024 topk API 结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-024-topk-api.md`

Git 基线：`a5d4d3b`

说明：本轮在包含 PLAN-021、PLAN-022、PLAN-023 未提交改动的工作区状态上继续实施。

## 实施结果

1. 新增顶层 `torch_1k.topk(input, k, dim=None, largest=True, sorted=True)`。
2. 新增 `Tensor.topk(k, dim=None, largest=True, sorted=True)`。
3. 返回 PyTorch 风格 `TopKResult(values, indices)`。
4. 支持默认最后一维。
5. 支持显式 `dim` 和负维度。
6. 支持 `largest=True` 和 `largest=False`。
7. 支持 `k=0` 返回空 values / indices。
8. values 支持反向传播：
   - 上游梯度通过 `put_along_axis` scatter 回输入对应位置。
9. indices 不追踪梯度：
   - 代码 review 时发现 indices 如果作为 `Function` 第二输出会被错误标记为可求导。
   - 已修正为 `TopK(Function)` 只产生可微 values，indices 作为独立 `requires_grad=False` Tensor 返回。
10. 新增 `examples/example24_topk_compare.py`：
    - 使用 top-k values / indices 计算 top-1 和 top-2 accuracy。
    - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
11. 更新设计文档，记录 PLAN-024 已补齐 topk API。

## 新增测试

新增 `tests/test_33_topk_api.py`，覆盖：

1. largest top-k 的 values 和 indices。
2. smallest top-k 和负维度。
3. 默认最后一维。
4. `sorted=False` 返回 top-k 集合，不约束顺序。
5. values 反向传播 scatter 到被选中位置。
6. `k=0` 空结果。
7. 可选 CUDA 路径下 values、indices 和梯度保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_33_topk_api.py
python examples/example24_topk_compare.py
USE_TORCH_1K=0 python examples/example24_topk_compare.py
pytest -q tests/test_30_loss_functional_reduction.py tests/test_31_shape_split_chunk_repeat.py tests/test_32_einsum_api.py tests/test_33_topk_api.py
python -m compileall -q torch_1k examples
pytest -q
```

结果：

- 新增 topk 测试：`7 passed`
- 新增 topk 示例：`torch_1k` 与 PyTorch 路径均通过
- PLAN-021 至 PLAN-024 新增测试集合：`26 passed`
- 全量测试：`156 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. `torch.sort` / `argsort` 核心子集已由 PLAN-026 补齐。
2. tie 场景未保证与 PyTorch 完全一致；不同后端排序稳定性可能不同。
3. `sorted=False` 当前仍返回有序 top-k 集合；PyTorch 对该模式不承诺顺序，因此不影响常用兼容。
4. 尚未实现 `out=` 参数。

## 复核结论

本轮补齐了分类评估、检索和 logits 调试中高频使用的 top-k 接口。它延续了 `max/argmax` 的预测索引 API 风格，同时保留 values 的梯度路径，适合教学展示“返回值可微、索引不可微”的算子边界。
