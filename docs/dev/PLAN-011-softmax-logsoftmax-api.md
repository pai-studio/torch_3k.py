# PLAN-011 Softmax / LogSoftmax API 兼容

日期：2026-06-24

Git 基线：`52e3212823d630e06061ee4b05af23cf32899470`

## 目标

补齐分类模型中高频使用的 Softmax / LogSoftmax API，让 `Tensor.softmax()`、`Tensor.log_softmax()`、`nn.functional.softmax()`、`nn.functional.log_softmax()`、`nn.Softmax` 和 `nn.LogSoftmax` 都能像 PyTorch 一样使用。示例继续保持只替换导入名即可运行。

## 当前缺口

1. 底层 `torch_1k.softmax` 已存在，但缺少 Tensor 方法形式。
2. `torch_1k.nn.functional` 缺少 `softmax` 和 `log_softmax`。
3. `torch_1k.nn` 缺少 `Softmax` 和 `LogSoftmax` 模块。
4. 设计文档仍记录 `Softmax` / `LogSoftmax` 暂未独立暴露。

## 实施步骤

1. 在 `torch_1k.functional.numeric` 中新增稳定 `log_softmax` Function。
2. 给 `Tensor` 增加 `softmax(dim)` 和 `log_softmax(dim)` 方法。
3. 在 `nn.functional` 中导出 `softmax` 和 `log_softmax`。
4. 新增 `nn.Softmax(dim=None)` 和 `nn.LogSoftmax(dim=None)` 模块。
5. 新增 `examples/example11_softmax_logsoftmax_compare.py`，覆盖 Tensor 方法、函数式 API 和模块 API。
6. 新增测试覆盖前向归一化、`exp(log_softmax)` 与 softmax 一致、反向传播梯度形状和双后端可运行。
7. 更新设计文档中 Softmax / LogSoftmax 状态。
8. 运行新增测试、双后端示例、全量测试和编译检查。

## 验收标准

1. `x.softmax(dim=1)` 和 `nn.functional.softmax(x, dim=1)` 可运行。
2. `x.log_softmax(dim=1)`、`nn.functional.log_softmax(x, dim=1)` 和 `nn.LogSoftmax(dim=1)` 可运行。
3. `exp(log_softmax(x))` 与 `softmax(x)` 数值一致。
4. `log_softmax` 反向传播可把梯度传回输入。
5. 新增示例在 `torch_1k` 与 PyTorch 下均通过。
6. 全量测试通过。
