# PLAN-016 BatchNorm 与 Module buffer 结果

日期：2026-06-24

计划文件：`docs/dev/PLAN-016-batchnorm-buffer.md`

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 实施结果

1. `Module` 新增最小 buffer 机制：
   - 新增 `_buffers`。
   - 新增 `register_buffer(name, tensor)`。
   - 新增 `named_buffers(prefix='')`。
   - `to(device)` 会迁移当前模块 buffer。
   - `state_dict()` 会保存参数和 buffer。
   - `load_state_dict()` 会恢复参数和 buffer。
2. 新增 `nn.BatchNorm1d`：
   - 支持 `(N, C)`。
   - 支持 `(N, C, L)`。
   - 支持 affine 参数。
   - 支持 running stats。
3. 新增 `nn.BatchNorm2d`：
   - 支持 `(N, C, H, W)`。
   - 支持 affine 参数。
   - 支持 running stats。
4. BatchNorm 训练态：
   - 使用当前 batch 的 mean/var 做归一化。
   - 更新 `running_mean` / `running_var`。
   - running var 更新使用 batch 无偏方差。
5. BatchNorm 推理态：
   - 使用 `running_mean` / `running_var`。
   - 不再使用当前 batch stats。
6. BatchNorm 前向主要复用现有 Tensor 算子组合，反向传播由 autograd 自动处理。
7. `torch_1k.nn.__init__` 已导出：
   - `BatchNorm1d`
   - `BatchNorm2d`
8. 新增 `examples/example16_batchnorm_compare.py`：
   - 覆盖 `train()` / `eval()` 行为差异。
   - 覆盖 affine 参数梯度。
   - 同一代码可通过 `USE_TORCH_1K=0` 切换到 PyTorch。
9. 更新设计文档：
   - 记录 BatchNorm 已补齐。
   - 记录 `Module` buffer 已支持 checkpoint 和设备迁移。

## 新增测试

新增 `tests/test_25_batchnorm.py`，覆盖：

1. `BatchNorm1d` 训练态按 channel 归一化。
2. affine 参数参与反向传播。
3. `running_mean` / `running_var` 更新。
4. eval 模式使用 running stats。
5. `BatchNorm2d` 对 `(N, C, H, W)` 按 channel 归一化。
6. BatchNorm buffer 进入 `state_dict()` 并可 `load_state_dict()` 恢复。
7. 嵌套模块的 `state_dict()` 包含 BatchNorm buffer。
8. 非法输入维度明确报错。
9. 可选 CUDA 路径下参数、buffer、输出和梯度保持 CUDA 设备。

## 验证结果

已运行：

```bash
pytest -q tests/test_25_batchnorm.py
python examples/example16_batchnorm_compare.py
USE_TORCH_1K=0 python examples/example16_batchnorm_compare.py
pytest -q
python -m compileall -q torch_1k examples
```

结果：

- 新增 BatchNorm 测试：`8 passed`
- BatchNorm 双后端示例：`torch_1k` 与 PyTorch 路径均通过
- 全量测试：`105 passed`
- 编译检查：`torch_1k` 与 `examples` 通过

## TODO / 未完成事项

1. 尚未实现 `BatchNorm3d`、`SyncBatchNorm` 或 lazy BatchNorm。
2. 尚未实现 `num_batches_tracked`。
3. 尚未覆盖 PyTorch 完整 dtype promotion 和 channels-last 优化。

## 复核结论

本轮补齐了真实 CNN/MLP 训练中高频使用的 BatchNorm 核心能力，并补上了 BatchNorm 必需的 Module buffer 状态管理。当前 `train()` / `eval()` 已能同时驱动 Dropout 和 BatchNorm，checkpoint 也能保存恢复参数外的 running stats。
