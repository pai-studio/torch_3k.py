# PLAN-035 min API 兼容

日期：2026-06-26

Git 基线：`c448c68`

## 背景

PLAN-012 已补齐 `torch.max` / `Tensor.max` 的常用路径，PLAN-033 和 PLAN-034 又补齐了 values-only 的 `amax` / `amin` 以及 `aminmax`。但 PLAN-034 结果文档仍记录一个相邻缺口：

1. 尚未实现 `torch.min` / `Tensor.min` 的 values/indices 语义。

在 PyTorch 常用代码中，`min` 与 `max` 成组出现：全局最小值用于数值检查，按维度最小值用于取 values/indices，elementwise min 用于裁剪、边界融合和数值保护。补齐 `min` 可以让当前规约 API 更对称，也减少用户从 PyTorch 迁移时的替换阻力。

## 目标

1. 新增顶层函数：
   - `torch_1k.min(input)`
   - `torch_1k.min(input, dim, keepdim=False)`
   - `torch_1k.min(input, other)`
   - `torch_1k.minimum(input, other)`
2. 新增 Tensor 方法：
   - `Tensor.min(dim=None, keepdim=False)`
3. 支持：
   - 全局最小值规约。
   - 单维最小值规约，返回 `values` / `indices`。
   - 负维度。
   - `keepdim=True`。
   - elementwise min。
4. 梯度语义与 PyTorch 对齐：
   - 全局 `min(input)` 对重复最小值平均分配梯度。
   - `min(input, dim=...)` 只向 argmin 位置回传梯度。
   - elementwise min 在相等位置对两个输入平均分配梯度。
5. 保持现有 `amax` / `amin` / `max` 路径不回归。
6. CPU/CUDA 后端一致。

## 实施步骤

1. 改造 `torch_1k/functional/numeric.py`：
   - 新增 `Minimum(Function)`。
   - 新增 `Min(Function)`。
   - 新增 `_min_indices(...)`。
   - 新增 `minimum(...)` 与 `min(...)`。
   - 新增 `MinResult(values, indices)` 返回类型。
2. 改造 `torch_1k/tensor.py`：
   - 新增 `Tensor.min(...)` 方法。
3. 新增测试：
   - 全局 `min` 重复最小值梯度与 PyTorch 对比。
   - `min(dim=...)` values/indices 与 argmin 梯度。
   - 负维度和 keepdim。
   - elementwise `min` / `minimum` 广播和 tie 梯度。
   - `Tensor.min(...)` 方法入口。
   - CUDA 可选路径。
4. 新增示例：
   - 对 logits / score 张量取每行最小值及索引。
   - 对两个张量做 elementwise min。
   - 同进程对比 `torch_1k` 与 PyTorch 的输出和梯度。
5. 更新 PLAN-034 结果文档和设计路线图。

## 非目标

1. 不实现 tuple dim，因为 PyTorch `torch.min(input, dim=...)` 只接受单个整数维度。
2. 不实现 `out=` 参数。
3. 不实现 named tensor dim。
4. 不实现完整 dtype promotion。

## 验收标准

1. 新增 min 测试通过。
2. 既有 max / amin / amax 相关测试不回归。
3. 新增示例与 PyTorch 对比通过。
4. 全量测试通过。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
