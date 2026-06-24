# PLAN-024 topk API

日期：2026-06-25

Git 基线：`a5d4d3b`

说明：当前工作区已包含 PLAN-021、PLAN-022、PLAN-023 的未提交改动，本计划在该状态上继续实施。

## 背景

分类、检索和模型调试中经常需要查看 logits 的 top-k 结果，例如 top-1 / top-5 accuracy。PLAN-013 已补齐 `argmax`，但结果文档仍记录 `topk` 和排序相关 API 未实现。

`topk` 对本库的教学价值也比较高：

- 前向返回 values 和 indices，能展示 PyTorch 风格的 namedtuple 返回。
- values 是可微的，反向传播需要把梯度 scatter 回被选中的位置。
- indices 不参与梯度，适合展示预测索引类 Tensor 的边界。
- CPU/CUDA 可共用 NumPy/CuPy 的排序能力。

## 目标

1. 新增顶层 `torch_1k.topk(input, k, dim=None, largest=True, sorted=True)`。
2. 新增 `Tensor.topk(k, dim=None, largest=True, sorted=True)`。
3. 返回 PyTorch 风格结果：
   - `values`
   - `indices`
4. 支持：
   - 默认最后一维。
   - 显式 `dim`。
   - 负维度。
   - `largest=True` 和 `largest=False`。
5. values 支持反向传播：
   - 上游梯度 scatter 回输入对应位置。
   - indices 不追踪梯度。
6. 保持 CPU/CUDA 后端一致。

## 实施步骤

1. 在 `torch_1k/functional/numeric.py` 中实现：
   - `TopKResult`
   - `TopK(Function)`
   - `topk(...)`
2. 在 `Tensor` 上新增 `topk(...)` 方法。
3. 新增测试：
   - top-k largest 前向 values/indices。
   - smallest 前向 values/indices。
   - 负维度。
   - values 反向梯度 scatter。
   - `Tensor.topk` 和顶层 `torch.topk`。
   - 可选 CUDA。
4. 新增 PyTorch 兼容示例：
   - 用 top-k 计算分类预测结果。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
5. 更新设计文档阶段记录。

## 非目标

1. 不实现完整 `torch.sort` / `argsort`。
2. 不保证 tie 场景与 PyTorch 完全一致；不同后端排序稳定性可能不同。
3. 不实现 `out=` 参数。
4. 不实现完整 dtype promotion。

## 验收标准

1. 新增 topk 测试通过。
2. 新增 topk 示例在 `torch_1k` 与 PyTorch 两条路径下通过。
3. PLAN-021 至 PLAN-024 新增测试集合通过。
4. 全量测试不回归。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
