# toJitModels - 模型转换工具

## 概述

`toJitModels` 目录包含了将训练好的PyTorch模型转换为JIT（Just-In-Time）格式的工具，用于模型部署和优化。

## 目录结构

```
toJitModels/
├── evalStructureModels/          # 模型结构定义
│   ├── Release.py               # 满血版模型
│   ├── Release_lightly.py       # 轻量版模型
│   ├── Release_lightly_extra.py # 极轻量版模型
│   └── Custom.py                # 自定义模型
├── 8.pth                        # 训练好的模型权重文件
└── toJitTrace.py               # 模型转换脚本
```

## 使用方法

### 1. 准备模型权重

将训练好的权重文件命名为 `8.pth` 并放置在 `toJitModels` 目录下，或修改脚本中的路径：

```python
model_path = "./8.pth"
```

### 2. 运行转换脚本

```bash
python toJitTrace.py
```

### 3. 导出结果

脚本运行后会输出模型名称，并生成对应的 `.pt` 文件：

```
Release_lightly
模型导出完成： ./Release_lightly.pt
```

## 支持的模型

- **Release**: 满血版模型，功能完整
- **Release_lightly**: 轻量版模型，性能平衡
- **Release_lightly_extra**: 极轻量版模型，速度优先
- **Custom**: 自定义模型，可配置参数

## 注意事项

1. 确保模型权重文件与对应的模型结构匹配
2. 转换前请备份原始模型文件
3. 转换后的 `.pt` 文件可用于生产环境部署

## 依赖要求

- PyTorch >= 1.10.0
- torchvision
- 相应的CUDA版本（如果使用GPU）

## 故障排除

如果转换失败，请检查：
- 模型权重文件是否存在且完整
- 模型结构定义是否正确
- PyTorch版本是否兼容