# PLAN-028 CrossEntropyLoss 高级常用参数

日期：2026-06-25

Git 基线：`81ffeb1`

## 背景

PLAN-021 已补齐 `CrossEntropyLoss` 的 reduction 和函数式入口，但结果文档仍记录三个高价值缺口：

- `weight`
- `ignore_index`
- `label_smoothing`
- 高维 `(N, C, ...)` 输入形态

这些能力在真实训练中非常常见：类别不均衡分类会使用 `weight`，NLP padding token 和语义分割 void label 会使用 `ignore_index`，现代分类训练常用 `label_smoothing`，图像分割则依赖 `(N, C, H, W)` logits。

PLAN-027 已补齐 `gather`、mask 和数值工具，本轮继续把这些能力落到损失函数层，提升完整训练链路的 PyTorch 替换兼容性。

## 目标

1. 扩展 `nn.CrossEntropyLoss` 参数：
   - `weight=None`
   - `ignore_index=-100`
   - `reduction="mean"`
   - `label_smoothing=0.0`
2. 扩展 `nn.functional.cross_entropy(...)` 同名参数。
3. 支持输入：
   - `(N, C)`
   - `(N, C, d1, d2, ...)`
4. 支持目标：
   - `(N,)`
   - `(N, d1, d2, ...)`
5. 支持 `reduction="none" | "sum" | "mean"`。
6. 对 `ignore_index`：
   - 前向忽略对应位置。
   - 反向对应 logits 梯度为 0。
   - `reduction="none"` 对忽略位置返回 0。
7. 对 `weight`：
   - 按目标类别加权 NLL。
   - `mean` 按非忽略目标类别权重和归一化。
8. 对 `label_smoothing`：
   - 覆盖常用标量平滑系数。
   - 与 `weight`、`ignore_index`、三种 reduction 组合可用。
9. 保持 CPU/CUDA 后端一致。

## 实施步骤

1. 改造 `torch_1k/nn/loss.py`：
   - 解析高维 logits 为二维 `(M, C)` 工作形态。
   - 保存还原梯度所需的输入形状和 target shape。
   - 实现稳定 `log_softmax`。
   - 实现 `weight` / `ignore_index` / `label_smoothing` 的 loss 和梯度。
2. 改造 `torch_1k/nn/functional.py`：
   - 扩展 `cross_entropy(...)` 签名。
3. 新增测试：
   - `ignore_index` 三种 reduction。
   - `weight` 的前向和梯度。
   - 高维 `(N, C, H, W)` 输入。
   - `label_smoothing` 与 PyTorch 对比。
   - 组合参数与 CUDA 可选路径。
4. 新增示例：
   - 分割风格 logits `(N, C, H, W)`。
   - 类别权重、ignore_index 和 label_smoothing。
   - 同进程对比 `torch_1k` 与 PyTorch 的前向和梯度。
5. 更新设计文档和 PLAN-021 结果文档中的过期 TODO。

## 非目标

1. 不支持 soft target / probability target 形式。
2. 不让 `weight` 参与反向传播。
3. 不实现完整 dtype promotion。
4. 不实现 `size_average` / `reduce` 旧参数。

## 验收标准

1. 新增测试通过。
2. 新增示例与 PyTorch 对比通过。
3. 既有 loss、训练链路和全量测试不回归。
4. `python -m compileall -q torch_1k examples` 通过。
5. 结果文档记录实现范围、验证结果和未完成事项。
