# PLAN-009 requires_grad 语义兼容

日期：2026-06-24

Git 基线：`c239444653d32363c70bcd78ea3f86753d93e5d7`

## 目标

补齐 PyTorch 风格的 `requires_grad` 语义，让公开创建函数默认不追踪输入数据梯度，只有显式 `requires_grad=True` 或 `nn.Parameter` 才参与反向传播。训练示例应继续只通过导入名切换在 `torch_1k` 与 PyTorch 下运行。

## 当前缺口

1. `torch.tensor(...)` 默认会建图，和 PyTorch 默认 `requires_grad=False` 不一致。
2. `Tensor` 没有 `requires_grad` 属性，也没有 `requires_grad_()` 原地开关。
3. `nn.Parameter` 只是空子类，没有明确默认 `requires_grad=True`。
4. `Function.__call__` 只看全局 `Config.enable_backprop`，没有判断输入是否需要梯度。
5. 训练输入数据会无谓积累梯度，DataLoader 输出 batch 也会无谓建图。

## 实施步骤

1. 给 `Tensor` 增加 `requires_grad` 属性与 `requires_grad_()` 方法。
2. 让 `torch.tensor`、`rand`、`randn`、`zeros`、`ones` 等公开创建函数默认 `requires_grad=False`，并支持显式参数。
3. 保持 `Tensor(...)` 直接构造在项目内部测试中的默认可求导行为，同时让 `ensure_tensor` 创建的常量不需要梯度。
4. 让 `nn.Parameter` 默认 `requires_grad=True`。
5. 改造 `Function.__call__`：
   - 只有 `Config.enable_backprop=True` 且至少一个输入 `requires_grad=True` 时才建立反向图。
   - 输出 Tensor 的 `requires_grad` 由上述条件决定。
6. 改造 `backward()`：
   - 只给 `requires_grad=True` 的输入累积梯度。
   - `torch.no_grad()` 内即使输入需要梯度也不建图。
7. 修正 `to()`、`detach()`、dtype 转换、state dict 等路径的 requires_grad 传播或断开语义。
8. 新增双后端 requires_grad 示例和回归测试。
9. 更新 README 与设计文档中 autograd 边界说明。

## 验收标准

1. `torch.tensor([1.0]).requires_grad` 默认为 `False`。
2. `torch.tensor([1.0], requires_grad=True)` 可正常反向传播并积累输入梯度。
3. `nn.Parameter(...)` 默认为 `requires_grad=True`。
4. 模型训练时输入数据默认不积累梯度，参数仍能积累梯度并更新。
5. `torch.no_grad()` 能阻止显式 requires_grad 输入建图。
6. 全量测试通过。
