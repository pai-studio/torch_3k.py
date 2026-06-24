# PLAN-003 PyTorch 兼容 MLP 训练示例

日期：2026-06-24

## 目标

先用 PyTorch 风格写一个 MLP 训练示例，再将导入替换为 `torch_1k`，要求示例无需改训练主体即可直接跑通。

## 实施步骤

1. 新增 MLP 分类训练示例，示例主体使用 PyTorch 常见接口。
2. 补齐示例所需的核心功能：
   - `nn.Sequential`
   - `nn.ReLU`
   - `nn.CrossEntropyLoss`
   - `torch.argmax`
   - `Tensor.mean()`、`Tensor.float()`、`Tensor.long()`、`Tensor.__eq__`
3. 修正反向传播和创建函数中的后端细节，使 CPU/CUDA 路径保持一致。
4. 增加回归测试，覆盖 MLP 训练、交叉熵、Sequential/ReLU。
5. 运行 PyTorch 路径、`torch_1k` 路径和全量测试。
6. 生成结果文档并提交推送。

## 验收标准

1. `python examples/example4_mlp_train_compare.py` 可直接跑通 `torch_1k` 路径。
2. `USE_TORCH_1K=0 python examples/example4_mlp_train_compare.py` 可直接跑通 PyTorch 路径。
3. MLP 训练后 XOR 分类准确率达到 1.0。
4. 全量测试通过。
