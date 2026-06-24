# PLAN-005 代码 review 与模型状态管理结果

日期：2026-06-24

## Review 修复

1. 修复 `model.eval()` 关闭 autograd 的问题：
   - `train_model()` / `eval_model()` 现在只切换 `Config.train`。
   - `torch.no_grad()` 仍然是唯一关闭建图的公开上下文。
2. 扩展 `Module.train(mode=True)` / `Module.eval()`：
   - 支持 `train(False)`。
   - 递归设置子模块 `training` 状态。
3. 将 `Module` 内部参数/子模块名容器从 `set` 改为稳定顺序列表，保证参数遍历和状态字典顺序可重复。
4. 从顶层导出 `torch_1k.float32`、`torch_1k.float64`、`torch_1k.long`。

## 下一个高价值目标实现

已实现模型状态管理：

1. `Module.named_parameters()`
2. `Module.state_dict()`
3. `Module.load_state_dict()`
4. `torch_1k.save()`
5. `torch_1k.load()`

`load_state_dict()` 在严格模式下会检查缺失和多余 key；加载参数时复制数组，避免模型参数与传入 state 共享内存。

这使训练后的模型可以保存、恢复，并支持用同一输入验证恢复前后的输出一致。

## 新增测试

新增 `tests/test_14_review_state.py`，覆盖：

1. `eval()` 后仍可反向传播。
2. `train/eval` 递归设置子模块状态。
3. dtype 常量顶层导出。
4. `state_dict` / `load_state_dict` / `save` / `load` 端到端恢复。

## 验证结果

已运行：

```bash
pytest -q tests/test_14_review_state.py
python examples/example6_transformer_train_compare.py
python examples/example5_mnist_cnn_train_compare.py
USE_REAL_MNIST=1 python examples/example5_mnist_cnn_train_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增测试：`4 passed`
- Transformer 示例：`accuracy=1.000000`
- CNN MNIST-like 示例：`accuracy=1.000000`
- 真实 MNIST 小子集示例：`accuracy=1.000000`
- 全量测试：`46 passed`
- 编译检查：`torch_1k` 与 `examples` 通过
