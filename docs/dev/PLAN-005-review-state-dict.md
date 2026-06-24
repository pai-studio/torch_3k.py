# PLAN-005 代码 review 与模型状态管理

日期：2026-06-24

## 目标

系统 review 当前代码，修复影响 PyTorch 兼容性和真实训练稳定性的问题；选择并实现下一个高价值目标。

## Review 发现

1. `model.eval()` 当前会关闭 `Config.enable_backprop`，这与 PyTorch 语义不一致。PyTorch 的 `eval()` 只切换训练/推理模式，不关闭 autograd。
2. `Module.train()` / `eval()` 没有递归设置子模块状态，也不支持 `train(False)`。
3. `Module` 用 `set` 保存参数和子模块名，遍历顺序不稳定，不利于可重复的参数遍历和状态保存。
4. `torch_1k.float32`、`torch_1k.float64`、`torch_1k.long` 已在 `tensor.py` 定义但未从顶层导出。

## 下一个高价值目标

在 CNN/MNIST 和 Transformer 已经能跑通后，真实训练工作流最缺的是模型状态管理：

1. `model.state_dict()`
2. `model.load_state_dict(...)`
3. `torch.save(...)`
4. `torch.load(...)`

这能支持训练后保存、重新加载、对比模型输出和迁移示例。

## 实施步骤

1. 修复 `train/eval` 语义，使其只设置训练状态，不影响 autograd。
2. 让 `Module` 递归维护 `training` 状态，并支持 `train(mode=True)`。
3. 将模块内部参数名容器改为稳定顺序。
4. 实现 `named_parameters()`、`state_dict()`、`load_state_dict()`。
5. 实现顶层 `torch_1k.save()` / `torch_1k.load()`。
6. 补充回归测试。
7. 运行示例、全量测试和编译检查。

## 验收标准

1. `model.eval()` 后仍可正常反向传播。
2. `model.train(False)` / `model.eval()` 递归设置子模块 `training=False`。
3. `state_dict/load_state_dict` 可恢复模型参数并得到一致输出。
4. `torch_1k.save/load` 可保存和加载 state dict。
5. 全量测试通过。
