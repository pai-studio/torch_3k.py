# PLAN-016 BatchNorm 与 Module buffer

日期：2026-06-24

Git 基线：`980956db792e074192b89adba0a48f59f3561820`

## 目标

补齐 CNN/MLP 真实训练中常用的 BatchNorm 核心子集，并为 `Module` 增加最小 buffer 机制，使 BatchNorm 的 `running_mean` / `running_var` 能随设备迁移和 checkpoint 保存恢复。

本计划聚焦教学型实现，不追求完整 PyTorch BatchNorm API。

## 当前缺口

1. 设计文档和历史结果中多次将 BatchNorm 记录为 `nn` 层剩余核心模块。
2. 当前 `model.train()` / `eval()` 已能驱动 Dropout，但缺少另一个真实训练中高频依赖 train/eval 差异的模块。
3. `Module.state_dict()` 目前只保存参数，不能保存 running stats 这类非参数状态。
4. `Module.to('cuda')` 目前只迁移参数，不能迁移 buffer。

## 实施步骤

1. 扩展 `Module`：
   - 增加 `_buffers` 列表。
   - 支持 `register_buffer(name, tensor)`。
   - `to(device)` 递归迁移参数和 buffer。
   - `state_dict()` 保存参数和 buffer。
   - `load_state_dict()` 恢复参数和 buffer。
2. 新增 `BatchNorm1d` / `BatchNorm2d`：
   - 支持 `num_features`、`eps`、`momentum`、`affine`、`track_running_stats`。
   - affine=True 时使用可训练 `weight` / `bias`。
   - track_running_stats=True 时维护 `running_mean` / `running_var`。
   - 训练态使用 batch mean/var 并更新 running stats。
   - 推理态使用 running stats。
3. BatchNorm 前向尽量使用现有 Tensor 算子组合，复用 autograd，而不是手写复杂 backward。
4. 在 `torch_1k.nn.__init__` 导出 `BatchNorm1d` 和 `BatchNorm2d`。
5. 新增 `examples/example16_batchnorm_compare.py`，覆盖 train/eval 与 PyTorch import 切换路径。
6. 新增 `tests/test_25_batchnorm.py`，覆盖：
   - 训练态输出归一化。
   - affine 参数反向传播。
   - running stats 更新。
   - eval 使用 running stats。
   - buffer 进入 state_dict / load_state_dict。
   - `to('cuda')` 可选路径迁移 buffer。
7. 更新设计文档记录 PLAN-016。
8. 运行新增测试、示例、全量测试和编译检查。

## 验收标准

1. `nn.BatchNorm1d(C)` 可处理 `(N, C)` 和 `(N, C, L)`。
2. `nn.BatchNorm2d(C)` 可处理 `(N, C, H, W)`。
3. 训练态会更新 `running_mean` / `running_var`。
4. 推理态会使用 running stats，而不是当前 batch stats。
5. affine 参数能参与反向传播。
6. `state_dict()` 包含 BatchNorm 参数和 running stats。
7. `Module.to('cuda')` 能迁移 BatchNorm buffer。
8. 全量测试通过。

## TODO 记录

1. 本计划不实现 `SyncBatchNorm`、`BatchNorm3d` 或 lazy BatchNorm。
2. 本计划不实现 PyTorch BatchNorm 的完整 dtype promotion 和 channels-last 优化。
3. `num_batches_tracked` 暂不纳入最小核心实现。
