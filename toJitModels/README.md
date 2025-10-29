# toJitModels - 模型转换工具

## 概述

`toJitModels` 目录包含了将训练好的 PyTorch 模型转换为 TorchScript（JIT，.pt）以及导出为 Rockchip RKNN（.rknn）的工具，用于模型部署与端侧推理。

## 目录结构

```
toJitModels/
├── evalStructureModels/          # 模型结构定义
│   ├── Release.py               # 满血版模型
│   ├── Release_lightly.py       # 轻量版模型
│   ├── Release_lightly_extra.py # 极轻量版模型
│   └── Custom.py                # 自定义模型
├── 8.pth                        # 训练好的模型权重文件（示例名）
├── toJitTrace.py               # 转换为 TorchScript/JIT 的脚本
├── toRknn.py                   # 转换为 RKNN 的脚本（需在虚拟机中执行）
├── calibration/                # 校准/量化图片目录（可选，存放代表性样本）
├── dataset.txt                 # 量化样本清单（每行一个图片路径）
└── 模型转换平台搭建.docx         # 平台搭建流程文档（可选）
```

## 使用方法（导出 TorchScript .pt）

### 1) 准备模型权重

将训练好的权重文件命名为 `8.pth` 并放置在 `toJitModels` 目录下，或修改脚本中的路径：

```python
model_path = "./8.pth"
```

### 2) 运行转换脚本

```bash
python toJitTrace.py
```

### 3) 导出结果

脚本运行后会输出模型名称，并生成对应的 `.pt` 文件：

```
Release_lightly
模型导出完成： ./Release_lightly.pt
```

## 使用方法（导出 RKNN .rknn｜需要虚拟机 + RKNN-Toolkit）

RKNN 导出需在“安装了 RKNN-Toolkit 的虚拟机环境”中运行 `toRknn.py`。原因：RKNN-Toolkit 对系统环境与依赖要求严格，且需与目标板（如 rk3576）的固件/驱动版本匹配。

> 提示：有关模型转换平台搭建的更详细操作可以参考《模型转换平台搭建.docx》

### A. 准备虚拟机（建议 Ubuntu 20.04/22.04）

1) 创建虚拟机（VMware/VirtualBox/WSL2 均可，推荐 VMware + Ubuntu 20.04/22.04）
2) 安装 Python 3.8（建议与本项目一致）
3) 安装依赖：
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
pip install numpy opencv-python
```
4) 安装 RKNN-Toolkit（必须从 Rockchip 官网下载与目标平台匹配的安装包）

- 前往 Rockchip 官方网站（开发者中心），根据目标平台与固件版本下载对应的 rknn-toolkit2 安装包（通常为 `.whl` 或压缩包，可能需要登录/申请权限）。
- 在虚拟机中离线安装下载的包。例如：
```bash
# 示例文件名，请以官网提供的实际文件为准
pip install ./rknn_toolkit2-<version>-cp38-cp38-linux_x86_64.whl
```
> 说明：rknn-toolkit2 一般不直接通过 PyPI 获取，请以官网发布版本为准，确保与目标板固件/驱动版本匹配。

### B. 准备导出素材

- 将以下文件拷贝到虚拟机内的 `toJitModels/` 目录：
  - 训练好的模型：`.pt` 或可被 `rknn.load_pytorch` 解析的 PyTorch 权重（示例：`v2_1_bg_500.pt`）
  - 量化样本清单：`dataset.txt`（每行一个图片路径，建议 100~1000 张具有代表性的图片）
  - 本仓库中的 `toRknn.py`

示例 `dataset.txt`：
```
/path/to/img_0001.jpg
/path/to/img_0002.jpg
...
```

### C. 配置 toRknn.py

关键参数（已在脚本中示例）：
- `model_path = '../v2_1_bg_500.pt'`：指向待导出的模型
- `input_size_list = [[1, 3, 512, 512]]`：输入形状（NCHW），需与训练/推理一致
- `rknn.config(target_platform="rk3576", mean_values=[255*0.485, 255*0.456, 255*0.406], std_values=[255*0.229, 255*0.224, 255*0.225])`
  - 请确保 mean/std 与训练阶段一致（若训练时是 [0,1] 归一化，请去掉乘 255 或做相应调整）
- `rknn.build(do_quantization=True, dataset='./dataset.txt')`：启用量化并指定样本清单
- `rknn.export_rknn('./v2_1_bg_500_ext.rknn')`：导出文件名

### D. 在虚拟机中执行导出

```bash
cd toJitModels
python toRknn.py
```

成功后将生成 `.rknn` 文件（示例：`v2_1_bg_500_ext.rknn`）。

### E. 可选：本地快速验证

`toRknn.py` 中已包含 `rknn.init_runtime()` 初始化，可在虚拟机内直接加载 RKNN 进行最小验证。若需要推理对比：
1) 解除脚本末尾的推理示例注释
2) 准备测试图片，保证与训练时同样的预处理
3) 对比 PyTorch 与 RKNN 输出（量化会引入轻微偏差）

### 常见问题与排查

- 加载失败（Load model failed!）
  - 检查 `model_path` 是否存在
  - `.pt` 是否可被 `rknn.load_pytorch` 正确解析；必要时提供原始模型结构（参考 `evalStructureModels/`）
- 构建失败（Build model failed!）
  - RKNN-Toolkit 版本与目标平台不匹配；请根据 Rockchip 文档更换版本
  - 输入尺寸/通道顺序不一致；减少 batch；确保 `dataset.txt` 可读
- 结果偏差大
  - mean/std 与训练不一致；输入范围（0~1 vs 0~255）不一致；量化样本分布不代表真实场景
- 运行时初始化失败（Init runtime environment failed!）
  - 检查固件/驱动与导出工具链匹配；固定 `target_platform`；在目标板端再次验证

## 支持的模型

- **Release**: 满血版模型，功能完整
- **Release_lightly**: 轻量版模型，性能平衡
- **Release_lightly_extra**: 极轻量版模型，速度优先
- **Custom**: 自定义模型，可配置参数

## 注意事项

1. 确保模型权重文件与对应的模型结构匹配
2. 转换前请备份原始模型文件
3. `.pt` 用于通用部署；`.rknn` 用于 Rockchip 端侧设备
4. RKNN 导出必须在安装了 RKNN-Toolkit 的虚拟机内进行，并与目标板版本匹配

## 依赖要求

- PyTorch >= 1.10.0
- torchvision
- 相应的 CUDA 版本（如果使用 GPU）
- RKNN-Toolkit（在虚拟机内安装，必须从 Rockchip 官网下载与目标板匹配的版本）
- OpenCV、NumPy（辅助预处理与验证）

## 故障排除

若导出失败，请优先检查：
- 模型权重与结构是否匹配；PyTorch 能否正常前向
- 预处理配置（mean/std、输入尺寸）与训练是否一致
- RKNN-Toolkit 与目标平台版本是否匹配；`dataset.txt` 是否准备充分