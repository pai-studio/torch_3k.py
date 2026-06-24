# PLAN-015 常用 shape API 兼容

日期：2026-06-24

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 目标

补齐真实 PyTorch 模型 forward 和数据预处理中高频使用的 shape API，使 `x.view(...)`、`x.unsqueeze(...)`、`x.squeeze(...)`、`x.flatten(...)` 和 `torch.flatten(x, start_dim=...)` 等写法可直接运行。

本计划只实现可由 `reshape` 表达的常见形状变换，不扩展高级索引或复杂 view 语义。

## 当前缺口

1. `Tensor` 已有 `reshape`，但缺少 PyTorch 常见别名 `view`。
2. 顶层有 `torch.unsqueeze`，但 `Tensor` 缺少 `x.unsqueeze(dim)`。
3. 缺少 `squeeze`，真实数据预处理经常需要去掉 batch/channel 中的单例维度。
4. `nn.Flatten` 内部自带一份 flatten shape 逻辑，但 `Tensor.flatten` 和 `torch.flatten` 不存在。
5. `functional.matrix` 中 `_Unsqueeze` 仍是未完成占位，实际顶层 `unsqueeze` 分散在 `misc.py`。

## 实施步骤

1. 在 `functional.matrix` 中实现：
   - `view(x, shape)`
   - `unsqueeze(x, dim)`
   - `squeeze(x, dim=None, axis=None)`
   - `flatten(x, start_dim=0, end_dim=-1)`
2. 上述 API 全部复用 `reshape`，使反向传播自动恢复原 shape。
3. 给 `Tensor` 增加：
   - `view(*shape)`
   - `unsqueeze(dim)`
   - `squeeze(dim=None, axis=None)`
   - `flatten(start_dim=0, end_dim=-1)`
4. 将 `misc.unsqueeze` 改为复用 `functional.unsqueeze`，避免重复逻辑。
5. 将 `nn.Flatten.forward` 改为调用 `x.flatten(...)`。
6. 新增 `examples/example15_shape_api_compare.py`，覆盖常见 forward reshape 写法。
7. 新增 `tests/test_24_shape_api.py`，覆盖：
   - `view` 与反向传播。
   - `unsqueeze` Tensor 方法与顶层函数。
   - `squeeze` 全局、指定 dim、负 dim。
   - `flatten` Tensor 方法、顶层函数和 `nn.Flatten`。
   - CUDA Tensor 可选路径。
8. 更新设计文档记录 PLAN-015。
9. 运行新增测试、示例、全量测试和编译检查。

## 验收标准

1. `x.view(batch, -1)` 可用且梯度回到原 shape。
2. `x.unsqueeze(1)` 与 `torch.unsqueeze(x, 1)` 可用。
3. `x.squeeze()` / `x.squeeze(dim)` 可用。
4. `x.flatten(start_dim=1)` 与 `torch.flatten(x, start_dim=1)` 可用。
5. `nn.Flatten` 与 `Tensor.flatten` 行为一致。
6. CUDA Tensor 经过 shape API 后仍保持 CUDA 设备。
7. 全量测试通过。

## TODO 记录

1. 本计划不实现 `as_strided` 或共享存储 view 语义；当前 Tensor 仍以教学型数组包装为主。
2. 本计划不实现高级索引、`cat`、`split`、`chunk` 等更大的 shape API 面。
