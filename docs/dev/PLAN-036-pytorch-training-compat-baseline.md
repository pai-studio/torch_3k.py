# PLAN-036 PyTorch 训练脚本替换兼容基线

日期：2026-06-26

Git 基线：`e7881de`

## 背景

前序计划已经补齐了大量单点 API：张量创建、shape、规约、索引、排序、`einsum`、损失函数、优化器参数组、DataLoader sampler、BatchNorm、Dropout、checkpoint 等。继续沿着单个函数补齐 `out=`、named tensor dim 或完整 dtype promotion，会逐渐偏离本项目“教学型 PyTorch 核心训练链路”的定位。

当前更高价值的目标是把这些零散能力收束为一个可运行、可回归的训练脚本兼容基线：真实训练主体保持 PyTorch 写法，只在顶部切换 `torch` / `torch_1k` 导入，即可完成数据加载、前向、损失、反传、优化、评估和 checkpoint。

## 目标

1. 新增一个覆盖多条真实训练链路的 PyTorch 兼容示例：
   - MLP 分类：`TensorDataset` / `DataLoader`、`Sequential`、`Linear`、`BatchNorm1d`、`Dropout`、`CrossEntropyLoss`、`AdamW`。
   - CNN 分类：`Conv2d`、`BatchNorm2d`、`MaxPool2d`、`Flatten`、`Linear`。
   - Transformer 分类：`Embedding`、`TransformerEncoderLayer`、`Linear`。
   - checkpoint：`state_dict()`、`load_state_dict()`、`torch.save()`、`torch.load()`。
2. 示例训练主体与 PyTorch 保持兼容：
   - 默认使用 `torch_1k`。
   - `USE_TORCH_1K=0` 时切换到 PyTorch。
   - 除导入块外，训练和评估主体不写 `torch_1k` 专属代码。
3. 新增测试，把该示例作为基线验收：
   - 直接运行 `torch_1k` 路径。
   - 在安装 PyTorch 的环境下运行 PyTorch 路径，证明同一主体可替换导入。
4. 修复示例暴露的系统性兼容缺口，但不追求完整 PyTorch API。
5. 更新 README 和设计路线图，关闭 README 中“只替换 torch 可做普通 MNIST 分类”这一大目标的未完成状态。

## 实施步骤

1. 新增 `examples/example36_pytorch_training_baseline.py`：
   - 使用同一套训练函数跑 MLP、CNN、Transformer。
   - 输出每条链路的 loss / accuracy。
   - 运行 checkpoint roundtrip，并校验恢复后输出一致。
2. 新增 `tests/test_42_pytorch_training_baseline.py`：
   - 导入示例模块并运行 `torch_1k` 基线。
   - 用 subprocess 运行 `USE_TORCH_1K=0` 路径；若未安装 PyTorch，则跳过该子项。
3. 按需修复兼容缺口：
   - 优先修真实训练主体会触发的问题。
   - 保持实现短小、可读，不引入复杂抽象。
4. 更新文档：
   - README 核心功能勾选训练脚本替换兼容基线。
   - roadmap 记录 PLAN-036 阶段更新。
   - 生成 `PLAN-036-pytorch-training-compat-baseline-OUTCOME.md`。

## 非目标

1. 不实现完整 `out=` 参数。
2. 不实现 named tensor dim。
3. 不实现完整 dtype promotion。
4. 不实现多进程 DataLoader、pinned memory 或 distributed sampler。
5. 不保证 PyTorch 与 `torch_1k` checkpoint 文件跨后端互相加载；本计划只验证同一后端内保存恢复。

## 验收标准

1. 新增示例在默认 `torch_1k` 路径通过。
2. 新增示例在 PyTorch 路径通过，或在未安装 PyTorch 时测试明确跳过。
3. 新增测试通过。
4. 全量测试通过。
5. `python -m compileall -q torch_1k examples` 通过。
6. README、roadmap 和 OUTCOME 记录实现范围、验证结果、最终 Git 记录和未完成事项。
