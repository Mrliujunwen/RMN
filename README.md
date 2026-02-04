# RMN (Relation Mamba Network)

关系Mamba网络：面向小样本学习的细粒度交叉相关建模与多专家特征合成

## 简介

RMN是一个用于小样本学习的深度学习模型，包含三个核心组件：

1. **MEFS (Multi-Expert Feature Synthesizer)**: 多专家特征合成器 - 实现从"被动提纯"到"主动合成"的原型构建
2. **BSC (Bidirectional Spatial Correspondence)**: 双向空间对应模块 - 捕获细粒度空间匹配关系
3. **CCMR (Cross-Correlation Mamba Reasoner)**: 交叉相关Mamba推理器 - 线性复杂度的序列关系推理

## 模型版本

### RMN-Base（基准版本）
- 仅包含 BSC + CCMR
- 不使用 MEFS
- 用于对比实验

### RMN-FixedWeight（消融实验）
- MEFS（固定权重）+ BSC + CCMR
- 验证多专家机制的有效性
- 专家通过固定权重（平均）融合

### RMN-SoftRouting（完整版，推荐）
- MEFS（软路由）+ BSC + CCMR
- 样本自适应的专家融合
- 实现"主动合成"最优原型

## 目录结构

```
rmn/
├── models/
│   ├── rmn.py              # RMN主模型
│   ├── mefs.py             # 多专家特征合成器
│   ├── bsc.py              # 双向空间对应模块
│   ├── ccmr.py             # 交叉相关Mamba推理器
│   ├── resnet.py           # ResNet骨干网络
│   └── dataloader/         # 数据加载器
├── common/                 # 工具函数
├── train_rmn.py            # 训练脚本
├── test_rmn.py             # 测试脚本
└── README.md               # 说明文档
```

## 使用方法

### 训练

#### 1. RMN-SoftRouting（完整版，推荐）

```bash
# 5-way 1-shot
python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 \
    -extra_dir rmn_soft_1shot -model_version soft \
    -fusion_method concat

# 5-way 5-shot
python train_rmn.py -dataset miniimagenet -way 5 -shot 5 -gpu 0 \
    -extra_dir rmn_soft_5shot -model_version soft \
    -fusion_method concat
```

#### 2. RMN-FixedWeight（消融实验）

```bash
# 5-way 1-shot
python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 \
    -extra_dir rmn_fixed_1shot -model_version fixed \
    -fusion_method concat
```

#### 3. RMN-Base（基准）

```bash
# 5-way 1-shot
python train_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 \
    -extra_dir rmn_base_1shot -model_version base \
    -fusion_method concat
```

### 测试

```bash
# 测试RMN-SoftRouting
python test_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 \
    -load_path checkpoints/rmn_soft_1shot/max_acc.pth \
    -model_version soft

# 测试RMN-FixedWeight
python test_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 \
    -load_path checkpoints/rmn_fixed_1shot/max_acc.pth \
    -model_version fixed

# 测试RMN-Base
python test_rmn.py -dataset miniimagenet -way 5 -shot 1 -gpu 0 \
    -load_path checkpoints/rmn_base_1shot/max_acc.pth \
    -model_version base
```

## 参数说明

### 模型参数

- `-model_version`: 模型版本
  - `soft`: RMN-SoftRouting（完整版，推荐）
  - `fixed`: RMN-FixedWeight（消融实验）
  - `base`: RMN-Base（基准版本）

- `-fusion_method`: 分支融合方式
  - `concat`: 拼接融合（推荐）
  - `adaptive`: 自适应融合
  - `add`: 加法融合

### MEFS参数（仅用于fixed和soft版本）

- `-num_experts`: 专家数量（默认: 3）
- `-router_hidden`: 路由网络隐藏层维度（默认: 128，仅用于soft版本）

### CCMR参数

- `-ccmr_hidden_dim`: 隐藏层维度（默认: 256）
- `-ccmr_d_state`: 状态维度（默认: 16）
- `-ccmr_n_layers`: Mamba层数（默认: 1-shot用2层，5-shot用3层）
- `-ccmr_dropout`: Dropout比率（默认: 0.1）
- `-cross_kernel`: 交叉相关核大小（默认: 5）

### BSC参数

- `-temperature_attn`: 注意力温度（默认: 5.0）
- `-temperature`: 相似度温度（默认: 0.2）

### 训练参数

- `-dataset`: 数据集（miniimagenet/tieredimagenet/cifar_fs/cub）
- `-way`: N-way（默认: 5）
- `-shot`: K-shot（默认: 1）
- `-query`: 每类查询样本数（默认: 15）
- `-gpu`: GPU编号
- `-max_epoch`: 训练轮数（默认: 80）
- `-lr`: 学习率（默认: 0.1）
- `-batch`: 批次大小（默认: 128）
## 许可证

MIT License
