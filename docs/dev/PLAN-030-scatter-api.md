# PLAN-030 scatter / scatter_add 常用 API

日期：2026-06-25

Git 基线：`81ffeb1`

说明：当前工作区已包含 PLAN-028 和 PLAN-029 的未提交改动，本计划在该状态上继续实施。

## 背景

PLAN-027 已补齐 `where`、`masked_fill`、`index_select`、`gather` 和常用数值工具，并在结果文档中明确记录仍未公开 `scatter` / `scatter_add` API。

当前内部已有 scatter-add 式梯度累加逻辑：

- `index_select.backward(...)` 用 `xp.add.at(...)` 把梯度回填到输入。
- `gather.backward(...)` 用 `xp.add.at(...)` 支持重复 index 的梯度累加。

但用户侧仍无法直接写 PyTorch 常见的 scatter 类代码，例如把 token / class / graph message 写回目标张量，或构造 one-hot / 累加桶。

## 目标

1. 新增顶层函数：
   - `torch_1k.scatter(input, dim, index, src)`
   - `torch_1k.scatter_add(input, dim, index, src)`
2. 新增 Tensor 方法：
   - `Tensor.scatter(dim, index, src)`
   - `Tensor.scatter_add(dim, index, src)`
3. 支持常用 Tensor `src` 形态：
   - `input` 与 `index` 同 rank。
   - `src` 与 `index` 同 shape。
   - `index` 中的值沿 `dim` 指向 `input` 对应位置。
4. 支持 CPU/CUDA 后端一致。
5. 支持 autograd：
   - `scatter_add` 对 input 的梯度为上游梯度本身。
   - `scatter_add` 对 src 的梯度为按 index 从上游梯度 gather。
   - `scatter` 对被覆盖的 input 位置梯度为 0，其余位置保留上游梯度。
   - `scatter` 对 src 的梯度为按 index 从上游梯度 gather。
6. 重复 index：
   - `scatter_add` 按累加语义处理。
   - `scatter` 保持底层数组写入语义，不承诺与 PyTorch 非确定重复写完全一致；测试只覆盖无重复 index。

## 实施步骤

1. 改造 `torch_1k/functional/numeric.py`：
   - 复用 `_normalize_axis(...)`、`_ensure_integer_index(...)`、`_broadcast_index_selector(...)`。
   - 新增 `Scatter` / `ScatterAdd` Function。
   - 增加 index 形状、src 形状和 index 值范围校验。
2. 改造 `torch_1k/tensor.py`：
   - 增加 `scatter(...)` 和 `scatter_add(...)` 方法。
3. 新增测试：
   - `scatter_add` 与 PyTorch 前向/梯度对比，覆盖重复 index。
   - `scatter` 与 PyTorch 前向/梯度对比，覆盖无重复 index。
   - Tensor 方法与顶层函数入口。
   - 非法 shape / index 报错。
   - CUDA 可选路径。
4. 新增示例：
   - 用 `scatter` 构造 one-hot。
   - 用 `scatter_add` 做按 index 累加桶。
   - 同进程对比 `torch_1k` 与 PyTorch。
5. 更新 PLAN-027 结果文档和设计路线图。

## 非目标

1. 不实现 `value=` 标量 overload。
2. 不实现 `reduce=` / `scatter_reduce`。
3. 不实现 `out=` 参数。
4. 不追求 `scatter` 在重复 index 下与 PyTorch 的非确定写入完全一致。
5. 不实现完整 dtype promotion。

## 验收标准

1. 新增 scatter 测试通过。
2. 既有 mask/index/gather 测试不回归。
3. 新增示例与 PyTorch 对比通过。
4. 全量测试通过。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
