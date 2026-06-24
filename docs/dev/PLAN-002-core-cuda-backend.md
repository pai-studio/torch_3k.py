# PLAN-002 核心定位与 CUDA 后端基础

日期：2026-06-24

## 目标

围绕新定位“最简代码实现 PyTorch 核心功能，整体约 3000 行以内，教学代码但核心稳定”，完成首批高价值实现：明确核心模块边界，并引入 CuPy/CUDA 后端基础。

## 实施步骤

1. 新增设计文档，说明新定位、真实训练核心链路、模块优先级、CUDA/CuPy 支持策略。
2. 先编写 PyTorch 兼容训练示例，要求训练主体不分叉，只切换顶部导入名。
3. 新增后端抽象模块，统一 NumPy/CuPy 数组创建、设备迁移和数组模块识别。
4. 扩展 `Tensor` 的设备能力：`device`、`to()`、`cpu()`、`cuda()`、`detach()`、CUDA Tensor 的 `numpy()` 转换。
5. 改造基础算子，避免写死 NumPy，使 CPU/CUDA 共享自动微分逻辑。
6. 修复影响真实训练稳定性的已知问题：ReLU 语法、Sigmoid 反向、Module.zero_grad、Tensor.randn、DataLoader 迭代协议、二元广播梯度。
7. 增加 CPU 回归测试与可选 CUDA 测试。
8. 运行示例和测试，记录结果，生成 OUTCOME 文档。

## 验收标准

1. CPU 主测试通过。
2. 无 CuPy 环境时 CUDA 测试可跳过，不影响整体测试。
3. 有 CuPy/CUDA 环境时 CUDA Tensor 基础计算和反传可用。
4. `Module.to('cuda')` 能递归迁移参数。
5. 文档、计划、结果文件随代码提交。
