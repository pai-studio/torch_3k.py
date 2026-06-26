# PLAN-030 scatter / scatter_add 常用 API 结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-030-scatter-api.md`

Git 基线：`81ffeb1`

## 实施结果

1. 新增顶层函数：
   - `torch_1k.scatter(input, dim, index, src)`
   - `torch_1k.scatter_add(input, dim, index, src)`
2. 新增 Tensor 方法：
   - `Tensor.scatter(dim, index, src)`
   - `Tensor.scatter_add(dim, index, src)`
3. 支持常用 Tensor `src` 形态：
   - `input` 与 `index` 同 rank。
   - `src.shape == index.shape`。
   - `index` 中的值沿 `dim` 指向 `input` 对应位置。
4. `scatter_add` 支持重复 index，按累加语义写入目标张量。
5. `scatter` 支持无重复 index 的常用写入路径。
6. 支持 autograd：
   - `scatter_add` 对 input 的梯度为上游梯度本身。
   - `scatter_add` 对 src 的梯度为按 index 从上游梯度 gather。
   - `scatter` 对被覆盖 input 位置的梯度为 0，其余位置保留上游梯度。
   - `scatter` 对 src 的梯度为按 index 从上游梯度 gather。
7. CPU/CUDA 共用同一套实现。

## 新增测试

新增 `tests/test_37_scatter_api.py`，覆盖：

1. `scatter_add` 重复 index 的前向与梯度和 PyTorch 对比。
2. `scatter` 无重复 index 的前向与梯度和 PyTorch 对比。
3. Tensor 方法与顶层函数入口一致。
4. `dim=0` 和负维度 `dim=-1`。
5. 非法 index rank、src shape 和 index 越界报错。
6. CUDA 可选路径下输出和梯度保持 CUDA 设备。

## 新增示例

新增 `examples/example30_scatter_compare.py`：

1. 用 `scatter` 构造 one-hot。
2. 用 `scatter_add` 做按 index 累加桶。
3. 同进程对比 `torch_1k` 与 PyTorch 的输出和梯度。

## 验证结果

已运行：

```bash
pytest -q tests/test_37_scatter_api.py
pytest -q tests/test_35_mask_indexing_numeric_api.py tests/test_37_scatter_api.py
pytest -q tests/test_35_mask_indexing_numeric_api.py tests/test_37_scatter_api.py tests/test_32_einsum_api.py tests/test_36_cross_entropy_advanced.py
python examples/example30_scatter_compare.py
python examples/example29_einsum_ellipsis_compare.py
python examples/example28_cross_entropy_advanced_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 scatter 测试：`6 passed`
- mask/index/numeric 与 scatter 相关测试：`14 passed`
- scatter、mask/numeric、einsum 和高级交叉熵相关测试：`47 passed`
- 新增 scatter 示例：通过，输出和梯度均与 PyTorch 一致
- 相邻功能示例：einsum ellipsis 与高级交叉熵示例均通过
- 全量测试：`203 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 `value=` 标量 overload。
2. 尚未实现 `reduce=` / `scatter_reduce`。
3. 尚未实现 `out=` 参数。
4. 不承诺 `scatter` 在重复 index 下与 PyTorch 的非确定写入完全一致。
5. 尚未实现完整 dtype promotion。

## 复核结论

本轮把 PLAN-027 中内部已有但未公开的 scatter-add 梯度模式提升为用户可用 API。现在 one-hot 构造、桶累加、按 index 写回等常见张量胶水代码可以使用 PyTorch 风格表达，并且和 `gather` / `index_select` 的梯度回填路径保持一致。
