# PLAN-013 argmax API 兼容

日期：2026-06-24

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 目标

补齐分类训练和推理代码中高频使用的 PyTorch 风格 `argmax` API，使 `logits.argmax(dim=1)`、`torch.argmax(logits, dim=1)` 和 `keepdim=True` 写法可直接运行。该能力用于预测类别索引，不参与反向传播。

## 当前缺口

1. 现有 `torch.argmax(input, dim=None)` 只支持最小顶层函数。
2. `Tensor` 缺少 `x.argmax()` / `x.argmax(dim=...)` 方法。
3. 顶层 `argmax` 不支持 `keepdim`。
4. 现有分类示例仍必须写 `torch.argmax(logits, dim=1)`，不能使用更常见的 Tensor 方法形式。
5. 缺少针对 CUDA 后端一致性的可选测试。

## 实施步骤

1. 在 functional 层实现设备感知的非可微 `argmax`：
   - 支持 `dim` / `axis` 参数。
   - 支持 `keepdim` / `keepdims` 参数。
   - 返回 int64 Tensor，且 `requires_grad=False`。
2. 将 `misc.argmax` 改为复用同一实现，保持顶层 API 行为一致。
3. 给 `Tensor` 增加 `argmax(dim=None, keepdim=False, axis=None, keepdims=None)` 方法。
4. 新增 `examples/example13_argmax_api_compare.py`，验证 Tensor 方法、顶层函数和 `keepdim`。
5. 新增 `tests/test_22_argmax_api.py`，覆盖：
   - 全局 argmax。
   - dim argmax。
   - keepdim shape。
   - Tensor 方法形式。
   - 非 Tensor 输入。
   - 可选 CUDA 输入。
6. 更新设计文档，把 `argmax` 从剩余规约/索引缺口中移出。
7. 运行新增测试、示例、全量测试和编译检查。

## 验收标准

1. `x.argmax()` 返回扁平化最大值位置。
2. `x.argmax(dim=1)` 返回每行最大值位置。
3. `x.argmax(dim=1, keepdim=True)` 保留被规约维度。
4. `torch.argmax(x, dim=1, keepdim=True)` 与 Tensor 方法一致。
5. 返回结果不追踪梯度。
6. CUDA 可用时，CUDA Tensor 上的结果仍在 CUDA 设备。
7. 全量测试通过。

## TODO 记录

1. 本计划不实现 `topk`、`argsort` 或可微排序。
2. 本计划不扩展 `argmax` 的 tuple 维度语义；PyTorch 常用分类路径只需要单个 `dim`。
