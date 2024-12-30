# 深度学习训练框架

这是一个基于 PyTorch Lightning 的深度学习训练框架，支持图像分类任务，提供了灵活的配置选项和实验管理功能。

## 主要特性

- 使用 PyTorch Lightning 构建，代码结构清晰
- 支持 Weights & Biases (wandb) 实验跟踪
- 灵活的配置系统，支持 YAML 配置和命令行参数
- 支持模型断点保存和恢复
- 内置数据可视化工具
- 支持混合精度训练
- 自动记录训练指标

## 配置说明

主要配置参数在 `configs/config.yaml` 中设置：

### 实验环境配置

- `seed`: 随机种子
- `exp_name`: 实验名称
- `project`: wandb 项目名称

### 数据集配置

- `data_path`: 数据集路径
- `image_size`: 输入图像大小
- `num_classes`: 分类类别数

### 模型配置

- `model_name`: 使用的模型架构
- `pretrained`: 是否使用预训练权重

### 训练配置

- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `epochs`: 训练轮数
- `precision`: 训练精度模式

## 使用方法

1. 安装依赖：
   `bash`
   pip install -r requirements.txt

2. 测试训练模型：
   `bash`
   python train.py

## 项目结构

.
├── configs/ # 配置文件
├── tools/ # 工具函数
│ ├── datasets/ # 数据集相关
│ ├── models/ # 模型定义
│ ├── losses/ # 损失定义
│ └── pl_tool.py # Lightning 模块
├── train.py # 训练脚本
└── test.py # 测试脚本
