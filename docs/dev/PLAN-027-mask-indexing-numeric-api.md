# PLAN-027 掩码、索引选择与数值工具 API

日期：2026-06-25

Git 基线：`9e68cd0`

## 背景

当前基础算子已覆盖训练主干、排序族和 `einsum`。下一批高价值缺口集中在真实 PyTorch 代码中的“胶水算子”：

- 条件分支和掩码：`where`、`masked_fill`、比较运算。
- 按索引取值：`gather`、`index_select`。
- 数值保护和常用一元函数：`abs`、`sqrt`、`clamp` / `clip`。

这些 API 在 attention mask、按标签选择 logits、ranking 后处理、梯度裁剪前处理、loss 数值保护和调试代码中非常高频。对教学实现也有价值：它们能展示“掩码不可微、被选择的 values 可微、反向 scatter-add 回原输入”的核心自动微分模式。

## 目标

1. 新增比较掩码：
   - `==`、`!=`、`<`、`<=`、`>`、`>=`
   - 返回 `requires_grad=False` 的 bool Tensor。
2. 新增条件选择：
   - `torch_1k.where(condition, input, other)`
   - values 对 `input` 和 `other` 可反传，`condition` 不可微。
3. 新增掩码填充：
   - `Tensor.masked_fill(mask, value)`
   - `torch_1k.masked_fill(input, mask, value)`
   - 输入梯度在 mask 为 True 的位置为 0。
4. 新增索引选择：
   - `torch_1k.index_select(input, dim, index)`
   - `Tensor.index_select(dim, index)`
   - 支持 1D index，反向 scatter-add 回原输入。
5. 新增 gather：
   - `torch_1k.gather(input, dim, index)`
   - `Tensor.gather(dim, index)`
   - 支持 PyTorch 常见的 index 与 input 同 rank 形态，反向 scatter-add。
6. 新增数值工具：
   - `torch_1k.abs` / `Tensor.abs`
   - `torch_1k.sqrt` / `Tensor.sqrt`
   - `torch_1k.clamp` / `Tensor.clamp`
   - `torch_1k.clip` / `Tensor.clip`
7. 保持 CPU/CUDA 后端一致，CUDA 不可用时相关测试跳过。

## 实施步骤

1. 在 `torch_1k/functional/numeric.py` 中实现：
   - 比较 helper。
   - `Abs`、`Sqrt`、`Clamp`。
   - `Where`、`masked_fill`。
   - `IndexSelect`、`Gather`。
2. 在 `Tensor` 上新增方法：
   - `abs`、`sqrt`、`clamp`、`clip`
   - `masked_fill`
   - `index_select`
   - `gather`
3. 在 `register_ops()` 中补齐比较运算符。
4. 新增测试：
   - 比较掩码和 `where` 广播梯度。
   - `masked_fill` 梯度。
   - `index_select` 重复索引 scatter-add。
   - `gather` 与 PyTorch 前向和梯度对比。
   - `abs`、`sqrt`、`clamp` 前向和梯度。
   - 可选 CUDA。
5. 新增示例：
   - 用 `masked_fill` 构造 attention mask。
   - 用 `gather` 按 label 取目标 logits。
   - 用 `clamp/sqrt/abs` 做数值保护。
   - 同进程对比 `torch_1k` 与 PyTorch。
6. 更新设计文档阶段记录。

## 非目标

1. 不实现完整高级索引语义。
2. 不实现 `scatter` / `scatter_add` 公开 API。
3. 不实现 `where(condition)` 单参数形式。
4. 不实现 `gather(..., sparse_grad=True)`。
5. 不实现 `out=` 参数。
6. 不实现完整 dtype promotion。

## 验收标准

1. 新增测试通过。
2. 新增示例与 PyTorch 对比通过。
3. 全量测试不回归。
4. `python -m compileall -q torch_1k examples` 通过。
5. 结果文档记录实现范围、验证结果和未完成事项。
