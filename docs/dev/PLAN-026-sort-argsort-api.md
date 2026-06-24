# PLAN-026 sort / argsort 排序 API

日期：2026-06-25

Git 基线：`a5d4d3b`

说明：当前工作区已包含 PLAN-021 至 PLAN-025 的未提交改动，本计划在该状态上继续实施。

## 背景

PLAN-024 已补齐 `topk`，但结果文档仍记录完整 `torch.sort` / `argsort` 未实现。排序是分类评估、检索排序、序列处理和模型调试中的高频基础能力，也和 `topk` 共享同一类教学重点：

- 前向返回排序后的 values 和对应 indices。
- values 可微，反向传播需要把上游梯度按 indices scatter 回原输入。
- indices / argsort 不可微，适合继续明确预测索引类 Tensor 的边界。
- CPU/CUDA 可复用 NumPy/CuPy 的 `argsort`、`take_along_axis` 和 `put_along_axis`。

## 目标

1. 新增顶层 `torch_1k.sort(input, dim=-1, descending=False, stable=False)`。
2. 新增 `Tensor.sort(dim=-1, descending=False, stable=False)`。
3. 新增顶层 `torch_1k.argsort(input, dim=-1, descending=False, stable=False)`。
4. 新增 `Tensor.argsort(dim=-1, descending=False, stable=False)`。
5. `sort` 返回 PyTorch 风格结果：
   - `values`
   - `indices`
6. 支持：
   - 默认最后一维。
   - 显式 `dim`。
   - 负维度。
   - `descending=True`。
   - `stable` 兼容签名。
7. values 支持反向传播：
   - 上游梯度按排序 indices scatter 回输入原位置。
   - indices / argsort 结果不追踪梯度。
8. 保持 CPU/CUDA 后端一致。

## 实施步骤

1. 在 `torch_1k/functional/numeric.py` 中实现：
   - `SortResult`
   - `Sort(Function)`
   - `sort(...)`
   - `argsort(...)`
2. 在 `Tensor` 上新增 `sort(...)` 和 `argsort(...)` 方法。
3. 新增测试：
   - 升序排序 values / indices。
   - 降序排序。
   - 负维度。
   - `Tensor.sort` 和顶层 `torch.sort`。
   - `argsort` 顶层和 Tensor 方法。
   - values 反向传播 scatter。
   - indices / argsort 不可微。
   - 可选 CUDA。
4. 新增 PyTorch 兼容示例：
   - 用 `sort` 排序 logits 得到预测排名。
   - 用 `argsort` 复用排序索引。
   - 对比 `torch_1k` 与 PyTorch 的 values、indices 和梯度。
5. 更新设计文档阶段记录。

## 非目标

1. 不实现 `out=` 参数。
2. 不保证 tie 场景与 PyTorch 在所有后端上完全一致；不同后端排序稳定性可能不同。
3. `stable` 只作为兼容签名接收，当前不承诺稳定排序语义。
4. 不实现完整 dtype promotion。

## 验收标准

1. 新增 sort / argsort 测试通过。
2. 新增 sort / argsort 示例在 `torch_1k` 与 PyTorch 两条路径下通过。
3. PLAN-021 至 PLAN-026 新增测试集合通过。
4. 全量测试不回归。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
