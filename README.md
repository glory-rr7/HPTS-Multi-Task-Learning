# Project README | 项目说明

HTR_Release 是一个基于深度学习的图像修复与去水印工具，使用PyTorch框架开发。该项目提供了多种预训练模型，能够有效去除图像中的水印、文字或其他不需要的元素，并智能修复被遮挡的区域。

## 1. 项目结构

```
HTR_Release/
├── config/                 # 配置文件
│   ├── config.yaml        # 主配置文件
│   └── config_utils.py    # 配置加载工具
├── models/                # 模型定义
│   ├── Release.py         # 满血版模型
│   ├── Release_lightly.py # 轻量版模型
│   ├── Release_lightly_extra.py # 极轻量版模型
│   └── Custom.py          # 自定义模型
├── dataloader/            # 数据加载器
├── loss/                  # 损失函数
├── data/                  # 运行时数据存储
├── datasets/              # 数据集目录
├── runnetworks.py         # 主运行脚本
├── test.py               # 测试脚本
├── utils.py              # 工具函数
└── environment.yml       # 环境依赖
```

## 2. 数据准备

### 训练数据格式
```
datasets/
└── truedata/
    ├── train/           # 训练集
    │   ├── images/      # 原始图像
    │   ├── masks/       # 掩码图像
    │   └── targets/     # 目标图像
    ├── val/             # 验证集
    └── test/            # 测试集
```

## 3. Models Overview | 模型概述
| 模型 | 版本 | 用途            |
|-------|------|---------------|
| **Release** | 满血版 | 所有功能开启，效果最佳   |
| **Release_lightly** | 轻量版 | 性能略微降低，时间大幅减少 |
| **Release_lightly_extra** | 极轻量版 | 时间开销进一步减少     |
| **Custom** | 自定义 | 按场景自由调参       |



---

## 4. Custom Model Parameters | Custom 模型参数
| 参数           | 类型/默认             | 合法范围                            | 说明         |
|--------------|-------------------|---------------------------------|------------|
| `base`       | `int`, **32**     | 8 的倍数 (建议16–64)                 | 通道宽度 / 参数量 |
| `refinement` | `bool`, **True**  | True / False                    | 开启精修复模块    |
| `ffp`        | `bool`, **True**  | True / False                    | 启用 FFP 模块  |
| `ppm`        | `bool`, **True**  | True / False                    | 启用 PPM 模块  |
| `down_sample` | `float`, **1.0**  | 0.5, 0.75, 1.0                  | 全局下采样比例    |
| `am`         | `str`, **ela**    | ela / eca / ca / cbam / se / sa | 注意力模块名称    |

---

## 5. Experiment Safety Checklist | 实验安全清单
1. **训练前务必备份 `config.yaml`。**  
2. 采用易懂文件名，例如 `config_day_on_20250718.yaml`。
3. 重新加载权重时，使用备份文件，确保当前配置与保存模型结构一致。

---

### Backup Example | 备份示例
.yaml
```
......
......  #other

custom:
    base: 48
    refinement: True
    ffp: False
    ppm: True
    down_sample: 1.0
    am: sa

......  #other
......
```

---


## 6. Experience | 实验复现
**Release**

    base: 64
    refinement: True
    ffp: True
    ppm: True
    down_sample: 1.0
    am: ela

**Release_lightly**

    base: 32
    refinement: True
    ffp: False
    ppm: False
    down_sample: 1.0
    am: ela

**Release_lightly_extra**

    base: 32
    refinement: False
    ffp: False
    ppm: False
    down_sample: 0.5
    am: ela


---


## 7. Quick Start | 快速开始
### 环境配置

**1. 安装依赖**
    
    conda env create -f environment.yml

**2. 激活环境**

    conda activate HTRNet

**3. 安装torch**  
torch无法用conda下载，所以中途会报错退出，**需要进入环境后手动安装torch**，可能会需要用梯子

    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html 


### 运行程序

**配置好config.yaml后**，运行runnetworks.py，会在data目录下生成运行时文件

    python runnetworks.py


## 8. 性能指标

项目支持多种评估指标：
- **PSNR**：峰值信噪比，数值越高越好
- **SSIM**：结构相似性，范围0-1，越接近1越好
- **L1 Loss**：平均绝对误差
- **MSE Loss**：均方误差


**注意**：首次使用建议先运行预测模式测试环境配置，确保所有依赖正确安装后再进行训练。