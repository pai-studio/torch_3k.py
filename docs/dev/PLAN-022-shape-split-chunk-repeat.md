# PLAN-022 split/chunk/repeat shape API

日期：2026-06-25

Git 基线：`a5d4d3b`

说明：当前工作区已包含 PLAN-021 的未提交改动，本计划在该工作区状态上继续实施。

## 背景

本库的新定位是用教学友好的最简代码实现 PyTorch 核心训练链路。现有 shape API 已覆盖 `view`、`reshape`、`unsqueeze`、`squeeze`、`flatten`、`cat` 等常用入口，但 PLAN-015 和 PLAN-018 的结果文档仍记录 `split`、`chunk`、`repeat` 未实现。

这些 API 在真实训练代码中常见：

- `x.split(...)`：按 batch、通道或序列维度拆分张量。
- `x.chunk(...)`：把一个张量按近似均匀大小拆成多段。
- `x.repeat(...)`：复制标签、位置、mask 或小张量到目标形状。

本轮补齐这组 shape API，优先保证常用 PyTorch 语义、自动微分和 CUDA/CuPy 后端一致性。

## 目标

1. 新增 `Tensor.repeat(*repeats)`。
2. 新增顶层 `torch_1k.repeat(input, *repeats)`。
3. 新增 `Tensor.split(split_size_or_sections, dim=0)`。
4. 新增顶层 `torch_1k.split(input, split_size_or_sections, dim=0)`。
5. 新增 `Tensor.chunk(chunks, dim=0)`。
6. 新增顶层 `torch_1k.chunk(input, chunks, dim=0)`。
7. 保持 split/chunk 的反向传播按切片回填。
8. 保持 repeat 的反向传播按重复维度求和还原到输入形状。
9. 覆盖可选 CUDA 路径。

## 实施步骤

1. 在 `torch_1k/functional/matrix.py` 中实现：
   - `Repeat(Function)`
   - `repeat(...)`
   - `split(...)`
   - `chunk(...)`
2. 在 `Tensor` 上新增同名方法：
   - `repeat`
   - `split`
   - `chunk`
3. 增加 PyTorch 兼容示例：
   - 同一文件通过 `USE_TORCH_1K=0/1` 切换实现。
4. 增加测试：
   - 覆盖 repeat 前向和反向。
   - 覆盖 split 按整数大小拆分。
   - 覆盖 split 按 sections 拆分。
   - 覆盖 chunk 非整除拆分。
   - 覆盖负维度。
   - 覆盖可选 CUDA。
5. 更新设计文档阶段记录。

## 非目标

1. 不实现 `repeat_interleave`。
2. 不实现共享存储 view 语义；当前仍保持教学型 Tensor 包装。
3. 不追求 PyTorch 对所有空张量边界的完整细节。
4. 不实现高级索引和复杂 mask 语义。

## 验收标准

1. 新增测试通过。
2. 新增示例在 `torch_1k` 与 PyTorch 两条路径下通过。
3. 现有全量测试不回归。
4. `python -m compileall -q torch_1k examples` 通过。
5. 结果文档记录实现范围、验证结果和未完成事项。
