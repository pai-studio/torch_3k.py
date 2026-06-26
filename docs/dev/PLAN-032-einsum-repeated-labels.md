# PLAN-032 einsum repeated labels / diagonal 常用路径

日期：2026-06-25

Git 基线：`81ffeb1`

说明：当前工作区已包含 PLAN-028 至 PLAN-031 的未提交改动，本计划在该状态上继续实施。

## 背景

PLAN-023 引入了 `torch.einsum` 核心子集，PLAN-025 加固了广播反向，PLAN-029 补齐了 ellipsis 常用路径。目前结果文档仍记录一个高价值缺口：不支持同一输入内部 repeated labels / diagonal，例如：

```python
torch.einsum("ii->i", x)   # diagonal
torch.einsum("ii->", x)    # trace
torch.einsum("ijj->i", x)  # 沿重复标签取对角后规约
```

这类写法常用于 trace、diagonal 抽取、矩阵二次型和紧凑张量表达，是 `einsum` 完整核心语义中的重要部分。

## 目标

1. 支持同一输入内部 repeated labels：
   - `"ii->i"`
   - `"ii->"`
   - `"ijj->i"`
   - 多输入组合中的 repeated labels，例如 `"ii,i->"`。
2. 与现有 ellipsis 展开兼容：
   - `"...ii->...i"`
   - `"...ii->..."`
3. 前向行为与 PyTorch 对齐：
   - 重复标签对应轴尺寸必须一致。
   - 隐式输出中重复标签不自动输出。
4. 反向传播：
   - 对 repeated label 产生的 diagonal / trace 路径，把梯度 scatter 回原输入对角线位置。
   - 非对角线位置梯度为 0。
   - 保持已有非 repeated-label einsum 反向逻辑不回归。
5. CPU/CUDA 后端一致。

## 实施步骤

1. 改造 `torch_1k/functional/matrix.py`：
   - 解析阶段不再拒绝同一输入内部 repeated labels。
   - expansion 阶段校验同一输入中重复标签轴尺寸一致。
   - 为 backward 构造每个输入的 unique-label spec。
   - 对 repeated-label 输入，将 unique-label 梯度 scatter 回原 shape 对角线。
2. 更新 `tests/test_32_einsum_api.py`：
   - 删除 `"ii->i"` 的 unsupported 断言。
   - 新增 repeated-label 前向与梯度 PyTorch 对比。
   - 覆盖 ellipsis + repeated labels。
   - 覆盖尺寸不匹配报错。
   - 覆盖 CUDA 可选路径。
3. 新增示例：
   - diagonal、trace、带 ellipsis 的 batch trace。
   - 同进程对比 `torch_1k` 与 PyTorch。
4. 更新 PLAN-023 / PLAN-025 / PLAN-029 结果文档中的过期 TODO。
5. 更新设计路线图。

## 非目标

1. 不实现 PyTorch sublist equation 格式。
2. 不实现 `out=` 参数。
3. 不实现完整 dtype promotion。
4. 不额外暴露独立 `diagonal` / `trace` API。

## 验收标准

1. 新增 repeated-label einsum 测试通过。
2. 既有 einsum、ellipsis、mask/scatter/nonzero 测试不回归。
3. 新增示例与 PyTorch 对比通过。
4. 全量测试通过。
5. `python -m compileall -q torch_1k examples` 通过。
6. 结果文档记录实现范围、验证结果和未完成事项。
