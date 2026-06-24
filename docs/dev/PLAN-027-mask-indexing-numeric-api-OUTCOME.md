# PLAN-027 掩码、索引选择与数值工具 API 结果

日期：2026-06-25

计划文件：`docs/dev/PLAN-027-mask-indexing-numeric-api.md`

Git 基线：`9e68cd0`

## 实施结果

1. 新增比较掩码：
   - `==`、`!=`、`<`、`<=`、`>`、`>=`
   - 返回 `requires_grad=False` 的 bool Tensor。
2. 新增 PyTorch 风格 Tensor 布尔语义：
   - 单元素 Tensor 可转 bool。
   - 多元素 Tensor 在布尔上下文中明确报错，避免静默按对象真值判断。
3. 新增条件选择：
   - `torch_1k.where(condition, input, other)`
   - `condition` 不可微，`input` 和 `other` 可按被选位置反传。
   - 支持广播梯度还原。
4. 新增掩码填充：
   - `Tensor.masked_fill(mask, value)`
   - `torch_1k.masked_fill(input, mask, value)`
   - 输入梯度在 mask 为 True 的位置为 0。
5. 新增索引选择：
   - `torch_1k.index_select(input, dim, index)`
   - `Tensor.index_select(dim, index)`
   - 支持 1D index，重复索引通过 scatter-add 累加梯度。
6. 新增 gather：
   - `torch_1k.gather(input, dim, index)`
   - `Tensor.gather(dim, index)`
   - 支持 index 与 input 同 rank 的常见形态，重复索引通过 scatter-add 累加梯度。
7. 新增数值工具：
   - `torch_1k.abs` / `Tensor.abs` / Python `abs(x)`
   - `torch_1k.sqrt` / `Tensor.sqrt`
   - `torch_1k.clamp` / `Tensor.clamp`
   - `torch_1k.clip` / `Tensor.clip`
8. 修复一个 review 中暴露的隐藏问题：
   - `torch_1k/functional/matrix.py` 中 `split` 原本使用未限定的 `sum(sections)`。
   - 模块内也定义了张量规约 `sum`，比较语义补齐后该路径会错误进入 Tensor 布尔上下文。
   - 已改为 `builtins.sum(sections)`，明确使用 Python 内置求和。
9. 新增 `examples/example27_mask_indexing_numeric_compare.py`：
   - 用 `masked_fill` 构造无效位置 mask。
   - 用 `gather` 按 label 取目标分数。
   - 用 `where` 做正值筛选。
   - 用 `clamp`、`sqrt`、`abs` 做数值保护。
   - 同进程对比 `torch_1k` 与 PyTorch 的前向和梯度。
10. 更新设计文档，记录 PLAN-027 已补齐掩码、索引选择与数值工具 API。

## 新增测试

新增 `tests/test_35_mask_indexing_numeric_api.py`，覆盖：

1. 比较运算生成不可微 bool mask。
2. Tensor 单元素 / 多元素布尔语义。
3. `where` 广播前向和梯度与 PyTorch 一致。
4. `masked_fill` 在 mask 位置阻断输入梯度。
5. `index_select` 重复索引的梯度累加与 PyTorch 一致。
6. `gather` 重复索引的梯度累加与 PyTorch 一致。
7. `abs`、`sqrt`、`clamp`、`clip` 的前向和梯度与 PyTorch 一致。
8. `gather` 非法 index rank 报错。
9. 可选 CUDA 路径下输出和梯度保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_35_mask_indexing_numeric_api.py
pytest -q tests/test_31_shape_split_chunk_repeat.py tests/test_35_mask_indexing_numeric_api.py
python examples/example27_mask_indexing_numeric_compare.py
python -m compileall -q torch_1k examples
pytest -q
```

结果：

- PLAN-027 新增测试：`8 passed`
- split 回归与 PLAN-027 测试集合：`14 passed`
- 新增示例：与 PyTorch 的 masked values、gather values、where values、数值保护结果和梯度误差均在浮点舍入范围内
- 全量测试：`183 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现完整高级索引语义。
2. 尚未公开 `scatter` / `scatter_add` API。
3. 尚未实现 `where(condition)` 单参数形式。
4. 尚未实现 `gather(..., sparse_grad=True)`。
5. 尚未实现 `out=` 参数。
6. 尚未实现完整 dtype promotion。

## 复核结论

本轮补齐了真实模型代码中高频出现的一组胶水算子。现在 attention mask、按标签 gather logits、条件选择、数值裁剪和常见一元数值保护都可以用 PyTorch 风格表达；梯度路径保持教学可读，核心模式是 mask 分流和索引 scatter-add。
