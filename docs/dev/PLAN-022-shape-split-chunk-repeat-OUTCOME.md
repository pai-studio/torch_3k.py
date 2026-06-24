# PLAN-022 split/chunk/repeat shape API 结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-022-shape-split-chunk-repeat.md`

Git 基线：`a5d4d3b`

说明：本轮在包含 PLAN-021 未提交改动的工作区状态上继续实施。

## 实施结果

1. 新增 `Tensor.repeat(*repeats)`。
2. 新增顶层 `torch_1k.repeat(input, *repeats)`。
3. 新增 `Tensor.split(split_size_or_sections, dim=0)`。
4. 新增顶层 `torch_1k.split(input, split_size_or_sections, dim=0)`。
5. 新增 `Tensor.chunk(chunks, dim=0)`。
6. 新增顶层 `torch_1k.chunk(input, chunks, dim=0)`。
7. `repeat` 使用 `Repeat(Function)` 实现：
   - 前向通过后端 `tile` 执行。
   - 反向把梯度 reshape 为“重复维度 + 原维度”后按重复维度求和。
   - 支持额外 leading 维度，例如 `x.repeat(2, 1)`。
8. `split` 和 `chunk` 复用已有 `get_item` 切片能力：
   - 自动继承切片梯度回填。
   - 支持整数 split size。
   - 支持 sections 列表。
   - 支持负维度。
9. 新增 `examples/example22_shape_split_chunk_repeat_compare.py`：
   - 覆盖 `repeat`、`chunk`、`split` 的组合使用。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
10. 更新设计文档，记录 PLAN-022 已补齐这组 shape API。

## 新增测试

新增 `tests/test_31_shape_split_chunk_repeat.py`，覆盖：

1. `repeat` 前向值和梯度累加。
2. `repeat` 增加 leading 维度。
3. `split(split_size, dim=...)` 的切片形状和梯度回填。
4. `split([sections], dim=-1)` 的负维度和分段梯度。
5. `chunk(chunks, dim=...)` 的近似均匀拆分。
6. 可选 CUDA 路径下 `repeat`、`chunk` 和反向传播保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_31_shape_split_chunk_repeat.py
python examples/example22_shape_split_chunk_repeat_compare.py
USE_TORCH_1K=0 python examples/example22_shape_split_chunk_repeat_compare.py
```

结果：

- 新增 shape API 测试：`6 passed`
- 新增 shape API 示例：`torch_1k` 与 PyTorch 路径均通过

完整回归将在 PLAN-023 `einsum` 实施完成后一并运行。

## TODO / 未完成事项

1. 尚未实现 `repeat_interleave`。
2. 空张量边界只覆盖最小可用路径，未追求 PyTorch 完整细节。
3. 尚未实现共享存储 view 语义；当前仍是教学型 Tensor 包装。

## 复核结论

本轮补齐了此前 shape API TODO 中最常用的一组拆分与重复接口。它们能支撑模型代码里的张量拆分、批量复制和简单布局变换，并保持自动微分路径清晰可读。
