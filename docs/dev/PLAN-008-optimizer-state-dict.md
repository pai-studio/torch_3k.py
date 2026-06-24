# PLAN-008 优化器状态保存与恢复

日期：2026-06-24

Git 基线：`6290dc357a5588585be2ddba989b3d078a4e4bf8`

## 目标

补齐优化器 checkpoint 能力，让 `model.state_dict()` 之外还能保存和恢复优化器内部状态。重点覆盖已有优化器 `SGD`、`MomentumSGD`、`Adam`、`AdamW`，使训练恢复后继续 step 的结果与未中断训练保持一致。

## 当前缺口

1. `Optimizer` 没有 `state_dict()` / `load_state_dict()` 公共协议。
2. `MomentumSGD` 的 `velocity`、`Adam/AdamW` 的 `m/v/t` 只能存在当前优化器实例中。
3. 状态内部使用 `id(parameter)` 作为 key，不能直接用于新模型实例恢复。
4. 现有保存/加载测试只覆盖模型参数，没有覆盖 optimizer checkpoint。

## 实施步骤

1. 在 `Optimizer` 基类中提供参数顺序映射、状态 Tensor 拷贝和加载辅助方法。
2. 为 `SGD` 保存/恢复 `lr`。
3. 为 `MomentumSGD` 保存/恢复 `lr`、`momentum` 和每个参数的 `velocity`。
4. 为 `Adam` 保存/恢复 `lr`、`betas`、`eps`、`weight_decay`、全局步数 `t` 和每个参数的 `m/v`。
5. 为 `AdamW` 保存/恢复 decoupled `weight_decay`，并保持 AdamW 的解耦衰减语义。
6. 新增测试，验证 Adam checkpoint 恢复后继续训练与未中断训练输出一致，并覆盖 SGD/MomentumSGD 基础状态。
7. 更新设计文档中 checkpoint 工作流状态。
8. 运行新增测试、全量测试和编译检查。

## 验收标准

1. 所有已有优化器都有 `state_dict()` 和 `load_state_dict()`。
2. Adam 状态恢复到新模型后，继续训练得到与原模型继续训练一致的输出。
3. 加载状态时能按目标参数 device 拷贝状态数组。
4. 全量测试通过。
