# PLAN-010 Dropout 与 train/eval 行为

日期：2026-06-24

Git 基线：`9f4acc33fc98b2980f081d6b80c6708853ba80ae`

## 目标

补齐 PyTorch 高频训练态模块 `nn.Dropout` 和函数式 `nn.functional.dropout`，让 `model.train()` / `model.eval()` 不只是状态标记，而是能影响真实前向行为。示例代码继续保持只通过导入名切换 `torch_1k` 与 PyTorch。

## 当前缺口

1. `Module.train()` / `eval()` 已递归设置 `training`，但现有模块几乎没有依赖该状态的行为。
2. `torch_1k.nn` 缺少 `Dropout`，常见 MLP/Transformer 代码替换时会失败。
3. `torch_1k.nn.functional` 缺少 `dropout`。
4. 当前测试没有覆盖训练态随机 mask、eval 恒等映射和 Dropout 反向传播。

## 实施步骤

1. 实现 `nn.functional.dropout(input, p=0.5, training=True, inplace=False)`：
   - 校验 `p` 在 `[0, 1]`。
   - `training=False` 或 `p=0` 时返回输入。
   - `p=1` 时返回 `input * 0`。
   - 训练态按 inverted dropout 生成 mask，并按 `1 / (1 - p)` 缩放。
2. 实现 `nn.Dropout(p=0.5, inplace=False)`，forward 使用模块 `self.training`。
3. 从 `torch_1k.nn` 导出 `Dropout`。
4. 新增 `examples/example10_dropout_train_eval_compare.py`，验证 train/eval 行为可在 PyTorch 与 `torch_1k` 双路径运行。
5. 新增测试覆盖训练态 mask、反向梯度、eval 恒等映射、函数式 dropout 和非法参数。
6. 更新设计文档中的 `train/eval` 行为状态。
7. 运行新增测试、双后端示例、全量测试和编译检查。

## 验收标准

1. `nn.Dropout` 在训练态会随机置零并缩放保留元素。
2. `nn.Dropout.eval()` 后输出等于输入。
3. Dropout 反向传播只通过保留元素并带缩放系数。
4. `nn.functional.dropout(..., training=False)` 返回恒等输出。
5. 新增示例在 `torch_1k` 与 PyTorch 下均通过。
6. 全量测试通过。
