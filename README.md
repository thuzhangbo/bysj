# 无源开放集域适应的图分类项目

## 项目概述

本项目实现了一个基于图神经网络（GNN）的无源开放集域适应（Source-Free Open-Set Domain Adaptation）框架，用于图分类任务。该框架能够在没有源域数据的情况下，仅利用预训练的源域模型和目标域数据，实现对新类别（未知类别）的识别和已知类别的分类。

### 核心特性

- **无源域适应**：无需访问源域数据，仅使用预训练模型
- **开放集识别**：能够识别目标域中的新类别（未知类别）
- **知识蒸馏**：使用教师-学生架构进行知识迁移
- **图神经网络**：基于GIN（Graph Isomorphism Network）的图分类模型
- **灵活配置**：支持多种数据集和超参数配置

## 项目架构

### 目录结构

```
bysj/
├── config.yaml              # 配置文件
├── requirements.txt          # 依赖包列表
├── data.py                  # 数据处理模块
├── model.py                 # 模型定义
├── utils.py                 # 工具函数
├── train_source.py          # 源域模型训练
├── adapt_target.py          # 目标域适应
├── evaluate.py              # 模型评估
├── extract_feature.py       # 特征提取
├── grid_train_source.py     # 网格搜索训练
├── test.py                  # 测试脚本
├── data/                    # 数据集目录
│   ├── ENZYMES/            # ENZYMES数据集
│   └── MUTAG/              # MUTAG数据集
├── checkpoints/             # 模型检查点
└── logs/                    # 训练日志
```

### 核心模块说明

#### 1. 数据处理模块 (`data.py`)
- **数据集加载**：支持TUDataset格式的图数据集
- **类别分配**：智能分配重叠类别样本到源域和目标域
- **标签映射**：创建连续标签映射，处理已知和未知类别
- **数据分割**：分层采样进行训练/验证/测试分割

#### 2. 模型架构 (`model.py`)
- **GIN模型**：基于图同构网络的图分类模型
- **特征提取**：支持中间特征提取功能
- **批量归一化**：提高训练稳定性
- **Dropout正则化**：防止过拟合

#### 3. 训练流程
- **源域训练** (`train_source.py`)：在源域数据上训练教师模型
- **目标域适应** (`adapt_target.py`)：使用知识蒸馏进行域适应
- **知识蒸馏**：教师模型指导学生模型学习已知类别

#### 4. 评估系统 (`evaluate.py`)
- **整体准确率**：评估所有类别的分类性能
- **已知类别准确率**：评估已知类别的识别能力
- **详细报告**：提供分类报告和混淆矩阵

## 技术原理

### 无源开放集域适应

1. **问题定义**：
   - 源域：包含已知类别 {0, 1, 2, 3}
   - 目标域：包含已知类别 {0, 1, 2, 3} + 未知类别 {4, 5}
   - 目标：在目标域上识别已知类别并检测未知类别

2. **解决方案**：
   - 使用教师-学生架构
   - 教师模型：在源域预训练，固定参数
   - 学生模型：在目标域训练，学习新类别

3. **知识蒸馏**：
   - 对已知类别样本使用KL散度损失
   - 对未知类别样本使用交叉熵损失
   - 平衡已知和未知类别的学习

### 图神经网络

使用GIN（Graph Isomorphism Network）作为基础模型：

```python
class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=5, dropout=0.5):
        # 多层GIN卷积 + 批量归一化 + 全局池化 + 分类头
```

## 环境配置

### 系统要求
- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖包
- `torch==2.5.0`：PyTorch深度学习框架
- `torch-geometric==2.6.1`：图神经网络库
- `scikit-learn`：机器学习工具
- `matplotlib`：可视化工具
- `PyYAML`：配置文件解析

## 使用方法

### 1. 配置设置

编辑 `config.yaml` 文件：

```yaml
# 数据集配置
dataset:
  name: "ENZYMES"                    # 数据集名称
  source_classes: [0, 1, 2, 3]      # 源域已知类别
  target_classes: [0, 1, 2, 3, 4, 5] # 目标域所有类别
  batch_size: 32

# 模型配置
model:
  name: "GIN"
  gin_hidden_dim: 32
  gin_layers: 4
  dropout: 0.1

# 训练配置
train:
  lr: 0.01
  epochs: 300
  weight_decay: 0.0
```

### 2. 源域模型训练

```bash
python train_source.py
```

训练过程：
- 加载数据集并分配类别
- 训练GIN模型
- 保存最佳模型到 `checkpoints/`

### 3. 目标域适应

```bash
python adapt_target.py
```

适应过程：
- 加载预训练的教师模型
- 初始化学生模型
- 使用知识蒸馏进行训练
- 保存适应后的模型

### 4. 模型评估

```bash
python evaluate.py
```

评估指标：
- 整体准确率
- 已知类别准确率
- 详细分类报告

### 5. 网格搜索（可选）

```bash
python grid_train_source.py
```

用于寻找最优超参数组合。

## 数据集

### 支持的数据集
- **ENZYMES**：酶分子分类数据集
- **MUTAG**：诱变分子数据集

### 数据集结构
```
data/
├── ENZYMES/
│   ├── processed/          # 处理后的数据
│   └── raw/               # 原始数据文件
└── MUTAG/
    ├── processed/
    └── raw/
```

### 自定义数据集
1. 将数据集放置在 `data/` 目录下
2. 修改 `config.yaml` 中的数据集配置
3. 确保数据格式符合TUDataset标准

## 实验结果

### 性能指标
- **整体准确率**：所有类别的分类准确率
- **已知类别准确率**：已知类别的识别准确率
- **训练曲线**：损失和准确率变化趋势

### 输出文件
- `checkpoints/`：模型检查点
- `logs/`：训练日志
- `train_curves.png`：训练曲线图
- `train_metrics.csv`：训练指标记录

## 配置参数详解

### 数据集配置
- `source_classes`：源域包含的已知类别
- `target_classes`：目标域包含的所有类别
- `source_ratio`：重叠类别中分配给源域的比例
- `batch_size`：批次大小

### 模型配置
- `gin_hidden_dim`：GIN隐藏层维度
- `gin_layers`：GIN层数
- `dropout`：Dropout比例

### 训练配置
- `lr`：学习率
- `epochs`：训练轮数
- `weight_decay`：权重衰减
- `early_stop_patience`：早停耐心值

### 域适应配置
- `method`：域适应方法
- `epochs`：适应轮数
- `threshold`：阈值参数

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小 `batch_size`
   - 减少 `gin_hidden_dim` 或 `gin_layers`

2. **数据集加载失败**
   - 检查数据集路径
   - 确保数据集格式正确

3. **模型收敛慢**
   - 调整学习率
   - 增加训练轮数
   - 检查数据分布

### 日志查看
训练日志保存在 `logs/` 目录下，包含详细的训练信息和错误信息。

## 扩展功能

### 支持新的域适应方法
1. 在 `adapt_target.py` 中添加新的损失函数
2. 修改训练循环以支持新方法
3. 更新配置文件

### 支持新的图神经网络
1. 在 `model.py` 中定义新的模型类
2. 更新配置文件中的模型参数
3. 修改训练脚本

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件

## 参考文献

1. Xu, K., et al. "How powerful are graph neural networks?" ICLR 2019.
2. Liang, J., et al. "Do we really need to access the source data? Source-free domain adaptation via distribution estimation." CVPR 2020.
3. Saito, K., et al. "Open-set domain adaptation by backpropagation." ECCV 2018.

---

**注意**：本项目仅供学术研究使用，请遵守相关数据集的使用协议。 