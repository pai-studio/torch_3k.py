# PLAN-011 Softmax / LogSoftmax API 兼容结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-011-softmax-logsoftmax-api.md`

Git 基线：`52e3212823d630e06061ee4b05af23cf32899470`

## 实施结果

1. 新增稳定 `torch_1k.log_softmax(x, dim=...)`：
   - 前向使用 shifted logits 和 log-sum-exp。
   - 反向使用 `gy - softmax(x) * sum(gy)` 标准公式。
2. 新增 Tensor 方法：
   - `x.softmax(dim=...)`
   - `x.log_softmax(dim=...)`
3. 新增 `torch_1k.nn.functional` API：
   - `nn.functional.softmax(x, dim=...)`
   - `nn.functional.log_softmax(x, dim=...)`
4. 新增模块：
   - `nn.Softmax(dim=...)`
   - `nn.LogSoftmax(dim=...)`
5. 新增 `examples/example11_softmax_logsoftmax_compare.py`：
   - 覆盖 Tensor 方法、函数式 API 和模块 API。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
6. 更新设计文档，移除 `Softmax` / `LogSoftmax` 未独立暴露的旧描述。

## 新增测试

新增 `tests/test_20_softmax_logsoftmax.py`，覆盖：

1. `Tensor.softmax(dim=1)` 和 `nn.functional.softmax(dim=1)` 行归一化。
2. `exp(log_softmax(x))` 与 `softmax(x)` 一致。
3. `log_softmax` 可反向传播到输入。
4. `nn.Softmax` / `nn.LogSoftmax` 与函数式 API 一致。

## 验证结果

已运行：

```bash
pytest -q tests/test_20_softmax_logsoftmax.py
python examples/example11_softmax_logsoftmax_compare.py
USE_TORCH_1K=0 python examples/example11_softmax_logsoftmax_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 Softmax/LogSoftmax 测试：`3 passed`
- Softmax/LogSoftmax 双后端示例：`torch_1k` 与 PyTorch 均通过
- 全量测试：`70 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## 复核结论

本轮补齐了分类输出后处理中最常见的一组 PyTorch API。当前 `dim=None` 会沿用底层默认最后一维，未实现 PyTorch 的 warning 行为；这不影响常规显式 `dim` 用法。后续高价值目标可以继续补齐 `max` / `Tensor.max` 或实现 BatchNorm。
