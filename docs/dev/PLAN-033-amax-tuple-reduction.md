# PLAN-033 amax 与多维 tuple 最大值规约

日期：2026-06-25

Git 基线：`81ffeb1`

说明：当前工作区已包含 PLAN-028 至 PLAN-032 的未提交改动，本计划在该状态上继续实施。

## 背景

PLAN-012 已补齐 `torch.max` / `Tensor.max` 的常用路径，包括全局 max、按单维 max、keepdim、values/indices 返回和 elementwise max。但结果文档仍记录两个高价值缺口：

1. 尚未实现 `torch.amax`。
2. 尚未实现多维 tuple 规约。

PyTorch 中 `torch.max(input, dim=...)` 返回 values/indices，适合单维分类推理；而 `torch.amax(input, dim=(...))` 返回 values-only，适合多维数值规约，例如图像空间维 `(H, W)`、序列多轴规约、mask 后的数值聚合等。

## 目标

1. 新增顶层函数：
   - `torch_1k.amax(input, dim=None, keepdim=False)`
2. 新增 Tensor 方法：
   - `Tensor.amax(dim=None, keepdim=False)`
3. 支持：
   - 全局规约。
   - 单维规约。
   - 多维 tuple/list 规约。
   - 负维度。
   - `keepdim=True`。
4. 梯度语义与 PyTorch `amax` 对齐：
   - 重复最大值位置平均分配梯度。
   - 多维规约时在所有最大值位置分配梯度。
5. 保持现有 `torch.max(input, dim=...)` 的 values/indices 单维语义不变。
6. CPU/CUDA 后端一致。

## 实施步骤

1. 改造 `torch_1k/functional/numeric.py`：
   - 新增 axis tuple 规范化工具。
   - 新增 `Amax(Function)`。
   - 新增 `amax(...)` 函数。
2. 改造 `torch_1k/tensor.py`：
   - 新增 `Tensor.amax(...)` 方法。
3. 新增测试：
   - 全局 `amax` 与重复最大值梯度。
   - tuple dim `amax(dim=(...))` 与 PyTorch 前向/梯度对比。
   - 负维度和 keepdim。
   - `torch.max(dim=...)` 旧路径不回归。
   - 非法重复维度 / 越界维度报错。
   - CUDA 可选路径。
4. 新增示例：
   - 对 `(N, C, H, W)` logits 做空间 `(H, W)` 最大值规约。
   - 同进程对比 `torch_1k` 与 PyTorch 的前向和梯度。
5. 更新 PLAN-012 结果文档和设计路线图。

## 非目标

1. 不改变 `torch.max(input, dim=...)` 的返回类型和 argmax 梯度语义。
2. 不实现 `torch.amin` / `torch.aminmax`。
3. 不实现 `out=` 参数。
4. 不实现完整 dtype promotion。

## 验收标准

1. 新增 amax 测试通过。
2. 既有 max、einsum、mask/index 测试不回归。
3. 新增示例与 PyTorch 对比通过。
4. 全量测试通过。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
