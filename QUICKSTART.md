# RMN 快速开始指南

## 环境配置

### 1. 创建Python环境

```bash
conda create -n rmn python=3.8
conda activate rmn
```

### 2. 安装依赖

```bash
# PyTorch (根据你的CUDA版本选择)
pip install torch torchvision torchaudio

# 其他依赖
pip install einops
pip install tqdm
pip install scipy
pip install pillow
```

### 3. 准备数据集

下载并配置数据集（miniImageNet, tieredImageNet, CIFAR-FS, CUB）：

```bash
# 创建数据目录
mkdir -p data

# miniImageNet (假设你已下载)
# 将数据放到 data/miniimagenet/ 目录下
```

## 验证安装

运行验证脚本检查模型是否正常：

```bash
python verify_rmn.py
```

期望输出：

```
============================================================
RMN (Relation Mamba Network) 集成测试
验证模型是否能正确实例化和运行
============================================================

[基准] 测试 RMN-Base（BSC + CCMR）
============================================================
测试 RMN-Base
============================================================
✓ 模型创建成功
  总参数量: 12,xxx,xxx
  可训练参数: 12,xxx,xxx
✓ 编码器前向传播成功
  输入形状: torch.Size([4, 3, 84, 84])
  输出形状: torch.Size([4, 640, 5, 5])
  特征增强模块: 未启用（Base版本）
✓ RMN-Base 测试通过

...（其他版本测试）...

✓ 所有测试通过！
```

## 训练模型

### 方式1: 基本训练（RMN-SoftRouting完整版）

```bash
# 5-way 1-shot
python train_rmn.py \
    -dataset miniimagenet \
    -way 5 \
    -shot 1 \
    -gpu 0 \
    -extra_dir rmn_soft_1shot \
    -model_version soft \
    -fusion_method concat

# 5-way 5-shot
python train_rmn.py \
    -dataset miniimagenet \
    -way 5 \
    -shot 5 \
    -gpu 0 \
    -extra_dir rmn_soft_5shot \
    -model_version soft \
    -fusion_method concat
```

### 方式2: 后台训练（使用tmux）

```bash
# 创建tmux会话并在后台运行
tmux new-session -d -s rmn_train '
python train_rmn.py \
    -dataset miniimagenet \
    -way 5 \
    -shot 1 \
    -gpu 0 \
    -extra_dir rmn_soft_1shot \
    -model_version soft \
    -fusion_method concat \
    > log_rmn_soft_1shot.log 2>&1
'

# 查看训练日志
tail -f log_rmn_soft_1shot.log

# 连接到tmux会话
tmux attach -t rmn_train
```

### 方式3: 消融实验

```bash
# 测试不同版本
# Base（基准）
python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 \
    -extra_dir rmn_base_1shot -model_version base

# FixedWeight（固定权重）
python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 \
    -extra_dir rmn_fixed_1shot -model_version fixed

# SoftRouting（软路由，完整版）
python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 \
    -extra_dir rmn_soft_1shot -model_version soft
```

## 测试模型

```bash
# 测试训练好的模型
python test_rmn.py \
    -dataset miniimagenet \
    -way 5 \
    -shot 1 \
    -gpu 0 \
    -load_path checkpoints/rmn_soft_1shot/max_acc.pth \
    -model_version soft
```

## 模型版本对比

| 版本 | MEFS | 融合方式 | 用途 |
|-----|------|---------|-----|
| Base | ✗ | - | 基准对比 |
| FixedWeight | ✓（固定权重） | 平均 | 消融实验：验证多专家有效性 |
| **SoftRouting** | ✓（软路由） | 样本自适应 | **完整方案（推荐）** |

## 预期性能

在 miniImageNet 数据集上的预期性能：

| 模型 | 5-way 1-shot | 5-way 5-shot |
|------|-------------|-------------|
| RMN-Base | 66.90 | 82.10 |
| RMN-FixedWeight | 66.85 | 82.42 |
| **RMN-SoftRouting** | **67.10** | **82.50** |

## 常见问题

### Q1: 训练很慢怎么办？

A: 检查以下几点：
- GPU是否正常工作（`nvidia-smi`）
- 减少batch大小（`-batch 64`）
- 减少worker数量（修改代码中的`num_workers`）

### Q2: 显存不足怎么办？

A: 
- 减少batch大小
- 使用梯度累积
- 减少CCMR层数（`-ccmr_n_layers 1`）

### Q3: 如何调整超参数？

A: 主要超参数：
- MEFS专家数量：`-num_experts 3`（推荐2-4个）
- CCMR层数：`-ccmr_n_layers 2`（1-shot用2，5-shot用3）
- 融合方式：`-fusion_method concat`（推荐concat）
- 学习率：`-lr 0.1`

## 下一步

1. 在其他数据集上训练（tieredImageNet, CIFAR-FS, CUB）
2. 尝试不同的融合策略
3. 调整专家数量和路由网络配置
4. 进行消融实验分析

## 技术支持

如有问题，请查看：
- README.md - 完整使用说明
- 代码注释 - 详细的实现说明
