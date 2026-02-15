# HTR_Release_2_DL — 手写/印刷体文字分离与图像修复

基于深度学习的文档图像处理框架，使用 PyTorch 开发。
核心能力：从混合文档图像中分离手写文字与印刷文字，并对被遮挡区域进行图像修复（去水印/去叠加文字）。

---

## 目录

1. [项目简介](#1-项目简介)
2. [项目结构](#2-项目结构)
3. [模型架构](#3-模型架构)
4. [数据格式](#4-数据格式)
5. [环境配置](#5-环境配置)
6. [配置文件说明](#6-配置文件说明)
7. [快速开始](#7-快速开始)
8. [模型变体](#8-模型变体)
9. [Custom 模型参数](#9-custom-模型参数)
10. [损失函数与评估指标](#10-损失函数与评估指标)
11. [模型部署与转换](#11-模型部署与转换)
12. [数据标注工具](#12-数据标注工具)
13. [实验安全清单](#13-实验安全清单)

---

## 1. 项目简介

HTR_Release_2_DL 针对以下场景设计：

- **手写/印刷体分离**：在手写与印刷内容混合的文档中，精确识别并分离两种文字类型
- **图像修复（Inpainting）**：重建被手写或印刷内容遮挡的原始区域
- **语义分割**：输出四类像素级分割掩码（背景、手写、印刷、重叠区域）

模型采用 **ResNet-UNet 混合架构**，支持多尺度输出、多种注意力机制，并提供从研究到边缘部署的完整工具链。

---

## 2. 项目结构

```
HTR_Release_2_DL/
├── config/
│   ├── config.yaml              # 主配置文件（训练/评估/预测参数）
│   └── config_utils.py          # 配置加载工具
├── models/
│   ├── Release.py               # 满血版模型（base=64）
│   ├── Release_lightly.py       # 轻量版模型（base=32）
│   ├── Release_lightly_extra.py # 极轻量版模型（base=16，0.5倍下采样）
│   ├── Custom.py                # 可自由调参的自定义模型
│   ├── networks.py              # 共用网络模块（FFP、PPM、注意力等）
│   └── __init__.py
├── dataloader/
│   ├── seg_datasets.py          # 标准图像数据集加载器
│   └── block_seg_datasets.py    # 分块（Tiled）数据集加载器（适合大图）
├── loss/
│   ├── Loss.py                  # 多任务损失函数及相似度指标
│   └── __init__.py
├── label_demo/                  # 数据标注工具（掩码生成、二值化）
├── toJitModels/
│   ├── toJitTrace.py            # 导出 TorchScript（.pt）
│   ├── toRknn.py                # 导出 RKNN（Rockchip 边缘设备）
│   └── README.md
├── runnetworks.py               # 主入口（训练 / 评估 / 预测）
├── utils.py                     # 图像处理、分块、增强等工具函数
├── environment.yml              # Conda 依赖定义
└── README.md
```

---

## 3. 模型架构

### 整体结构：ResNet-UNet 混合

```
输入图像 (3ch)
    │
    ├─ Encoder (ResNet 4级下采样 + 残差块)
    │       └─ PPM（金字塔池化，可选）
    │
    ├─ Decoder (对称上采样 + Skip Connection)
    │       ├─ FFP（特征融合金字塔，可选）
    │       └─ ELA / CA / CBAM / ECA / SA / SE 注意力模块
    │
    ├─ 多尺度输出：x_o1（1/8分辨率）、x_o2（1/4分辨率）、x_o3（原始分辨率）
    │
    └─ Refinement 精修复模块（可选）：膨胀卷积（dilation 2/4/8/16）
            ├─ output：最终修复图像（3ch RGB）
            └─ mask：语义分割（4类）
```

### 关键网络模块

| 模块 | 说明 |
|------|------|
| **PPM** | 金字塔池化，在最深层捕获全局多尺度上下文 |
| **FFP** | 特征融合金字塔，4级级联的并行特征精炼 |
| **Refinement** | 串行精修复，膨胀卷积扩大感受野，输出最终结果 |
| **ELA** | Efficient Layer Attention（默认） |
| **CA** | Coordinate Attention |
| **CBAM** | Convolutional Block Attention Module |
| **ECA** | Efficient Channel Attention |
| **SA** | Spatial Attention |
| **SE** | Squeeze-and-Excitation |

---

## 4. 数据格式

### 目录结构

```
datasets/
├── train/
│   ├── images/    # 输入：含水印或叠加文字的原始图像（RGB）
│   ├── rebuild/   # 目标：干净的参考图像（Ground Truth）
│   └── mask/      # 标注：语义分割掩码（4类彩色编码）
├── val/
│   ├── images/
│   ├── rebuild/
│   └── mask/
└── test/
    ├── images/
    ├── rebuild/
    └── mask/
```

### 掩码颜色编码

标注时使用 RGB 彩色掩码，颜色与类别的对应关系如下：

| 颜色 | RGB 值 | 类别 | 说明 |
|------|--------|------|------|
| 白色 | (255, 255, 255) | Class 0 | 背景 |
| 绿色 | (0, 255, 0) | Class 1 | 手写文字 |
| 红色 | (255, 0, 0) | Class 2 | 印刷文字 |
| 蓝色 | (0, 0, 255) | Class 3 | 手写与印刷重叠区域 |

### 预处理：mask → labels

训练时 dataloader 不直接读取彩色 mask，而是读取预处理后的灰度 label 图（像素值为 0~3），这样可以跳过运行时的颜色匹配，大幅加快数据加载速度。

使用 `temp/visualizer.py` 进行一次性预处理：

```python
from temp.visualizer import Visualizer

v = Visualizer()
v.preprocess_masks_to_labels("./datasets/train")
v.preprocess_masks_to_labels("./datasets/val")
v.preprocess_masks_to_labels("./datasets/test")
```

执行后会在每个数据集目录下生成 `labels/` 子目录，包含同名的灰度 PNG 图。最终目录结构如下：

```
datasets/train/
├── images/
├── rebuild/
├── mask/       # 原始彩色掩码（标注用，训练时不读取）
└── labels/     # 预处理后的灰度标签（训练时读取）
```

---

## 5. 环境配置

### 安装依赖

```bash
# 创建 conda 环境
conda env create -f environment.yml

# 激活环境
conda activate HTRNet
```

### 手动安装 PyTorch

> conda 无法自动解析 torch，需在激活环境后手动安装（需要网络或代理）

```bash
# CUDA 11.3
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 或更新版本
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html
```

**主要依赖：** PyTorch 1.10+、torchvision、scikit-image、opencv-python、Pillow、numpy、PyYAML

---

## 6. 配置文件说明

编辑 `config/config.yaml`，控制所有运行参数：

```yaml
run:
  type: 'train'           # 运行模式：train / evaluate / predict
  model: 'Release'        # 模型：Release / Release_lightly / Release_lightly_extra / Custom
  seed: 1                 # 随机种子（False 则不固定）
  data_crop:
    use: True
    size: 512             # 分块尺寸（需为128的倍数）

train:
  epochs: 50
  batch_size: 4
  learning_rate: 0.0002
  pretrained:
    use: True
    model_path: './data/train/.../models/60.pth'
  dataset_path: './datasets/train'
  numberworks: 8          # DataLoader 线程数

validation:
  dataset_path: './datasets/val'
  batch_size: 4

evaluate:
  model_path: './models/checkpoint.pth'
  dataset_path: './datasets/test'

predict:
  model_path: './models/model.pth'
  input_file_path: './datasets/image.jpg'

# 仅 model='Custom' 时生效
custom:
  base: 32
  refinement: True
  ffp: True
  ppm: True
  down_sample: 1.0
  am: 'ela'
```

---

## 7. 快速开始

### 训练

```bash
# config.yaml 中设置 run.type = 'train'
python runnetworks.py
```

运行后，日志与检查点保存在 `./data/train/{run_id}/` 下：
- `logs.csv`：每 epoch 的 loss、PSNR 等指标
- `models/`：检查点权重文件
- `samples/`：可视化验证样本（5×5 网格）

### 评估

```bash
# config.yaml 中设置 run.type = 'evaluate'
# 并填写 evaluate.model_path 和 evaluate.dataset_path
python runnetworks.py
```

输出每张图像及整体的 L1、MSE、PSNR、SSIM 指标，保存为 CSV。

### 单图预测

```bash
# config.yaml 中设置 run.type = 'predict'
# 并填写 predict.model_path 和 predict.input_file_path
python runnetworks.py
```

输出：`x1.png`、`x2.png`、`x3.png`、`output.png`、`mask.png`（中间层与最终结果）

> **提示**：首次使用建议先运行预测模式验证环境，再进行完整训练。

---

## 8. 模型变体

| 模型 | Base 通道数 | Refinement | FFP | PPM | 下采样 | 适用场景 |
|------|------------|-----------|-----|-----|--------|---------|
| **Release** | 64 | ✓ | ✓ | ✓ | 1.0 | 效果最佳，资源充足时使用 |
| **Release_lightly** | 32 | ✓ | ✗ | ✗ | 1.0 | 性能与速度均衡 |
| **Release_lightly_extra** | 32 | ✗ | ✗ | ✗ | 0.5 | 最快推理速度，轻度质量损失 |
| **Custom** | 可配置 | 可配置 | 可配置 | 可配置 | 可配置 | 研究与实验 |

---

## 9. Custom 模型参数

| 参数 | 类型 / 默认值 | 合法范围 | 说明 |
|------|--------------|---------|------|
| `base` | `int`, **32** | 8 的倍数（建议 16–64） | 通道宽度，决定模型参数量 |
| `refinement` | `bool`, **True** | True / False | 是否启用串行精修复模块 |
| `ffp` | `bool`, **True** | True / False | 是否启用特征融合金字塔 |
| `ppm` | `bool`, **True** | True / False | 是否启用金字塔池化模块 |
| `down_sample` | `float`, **1.0** | 0.5 / 0.75 / 1.0 | 全局下采样比例 |
| `am` | `str`, **ela** | ela / eca / ca / cbam / se / sa | 注意力机制选择 |

### 预置实验配置（可复现官方模型）

```yaml
# Release
base: 64 | refinement: True | ffp: True | ppm: True | down_sample: 1.0 | am: ela

# Release_lightly
base: 32 | refinement: True | ffp: False | ppm: False | down_sample: 1.0 | am: ela

# Release_lightly_extra
base: 32 | refinement: False | ffp: False | ppm: False | down_sample: 0.5 | am: ela
```

---

## 10. 损失函数与评估指标

### 多任务损失（训练）

```
总损失 = 精修复损失 + 掩码分割损失 + 多尺度重建损失
```

重建损失按语义类别加权（重叠区域权重最高）：

| 类别 | 权重 |
|------|------|
| 背景（Class 0） | 2 |
| 手写（Class 1） | 8 |
| 印刷（Class 2） | 10 |
| 重叠（Class 3） | 12 |

掩码分割使用交叉熵损失，类别权重为 (1.0, 4.0, 5.0, 6.0)。

### 评估指标

| 指标 | 说明 | 越好 |
|------|------|------|
| **PSNR** | 峰值信噪比（dB） | 越高越好 |
| **SSIM** | 结构相似性（0–1） | 越接近 1 越好 |
| **L1 Loss** | 平均绝对误差 | 越低越好 |
| **MSE Loss** | 均方误差 | 越低越好 |

---

## 11. 模型部署与转换

### 导出 TorchScript（跨平台 / C++ 推理）

```bash
cd toJitModels
python toJitTrace.py
# 输出：Release_lightly.pt
```

### 导出 RKNN（Rockchip 边缘设备）

```bash
cd toJitModels
python toRknn.py
# 输出：model.rknn
```

> 需要：Ubuntu 20.04/22.04、rknn-toolkit2（从 Rockchip 官方获取）
> 支持量化及校准集，适配 rk3576 等芯片

---

## 12. 数据标注工具

`label_demo/` 提供完整的训练数据制作流程：

| 脚本 | 功能 |
|------|------|
| `interactive_binarize.py` | 交互式图像二值化 UI |
| `single_binarize.py` | 单张图像二值化 |
| `batch_mask.py` | 批量生成语义掩码 |
| `batch_rebuild.py` | 批量生成 Ground Truth |
| `init_dir.py` | 初始化数据集目录结构 |

**标注流程：** 原始图像 → 二值化 → 手动绘制彩色区域掩码 → 生成 rebuild 目标图 → 整理为标准数据集目录

详见 `label_demo/README.md`。

---

## 13. 实验安全清单

1. **训练前备份 `config.yaml`**，建议使用语义化文件名，如 `config_release_20250718.yaml`
2. **加载预训练权重时**，确认当前配置与保存模型的结构完全一致（使用 `strict=False` 加载时注意缺失层）
3. **大图处理**：启用 `data_crop.use: True`，分块尺寸设为 512（需为 128 的倍数）
4. **多 GPU**：确认 CUDA 版本与 PyTorch 版本匹配
5. **首次运行**：建议先用 `predict` 模式验证环境与模型加载是否正常
