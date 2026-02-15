#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image
import torch

class Visualizer:
    """
    负责向量图片可视化，颜色<->类别的双向映射，以及 mask 批量预处理。
    """
    # 和 dataloader 中一致的均值/方差
    IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    def __init__(self,
                 color_to_label: Dict[Tuple[int, int, int], int] = None,
                 verbose: bool = True):
        if color_to_label is None:
            # 颜色到类别的映射
            color_to_label = {
                (255, 255, 255): 0,  # white -> 0 background
                (0, 255, 0): 1,      # green -> 1 printed
                (255, 0, 0): 2,      # red -> 2 handwriting
                (0, 0, 255): 3,      # blue -> 3 overlaped
            }
        self.color_to_label: Dict[Tuple[int, int, int], int] = color_to_label
        self.label_to_color: Dict[int, Tuple[int, int, int]] = {
            v: k for k, v in color_to_label.items()
        }
        self.verbose = verbose

    # ================= 单张图像转换 =================

    def rgb_mask_to_gray_label(self, mask_img: Image.Image) -> np.ndarray:
        """
        输入一张 RGB 掩码图（四种颜色），输出灰度标签图（值为 0~3）。

        Args:
            mask_img: PIL.Image，模式任意，会统一转成 RGB

        Returns:
            label_array: np.ndarray, dtype=uint8, shape=(H, W)，像素值 ∈ {0,1,2,3}
        """
        mask_img = mask_img.convert("RGB")
        mask_array = np.array(mask_img)  # (H, W, 3), uint8
        h, w, _ = mask_array.shape

        label_array = np.zeros((h, w), dtype=np.uint8)
        matched = np.zeros((h, w), dtype=bool)

        for (r, g, b), label in self.color_to_label.items():
            cond = (
                (mask_array[:, :, 0] == r) &
                (mask_array[:, :, 1] == g) &
                (mask_array[:, :, 2] == b)
            )
            label_array[cond] = label
            matched |= cond

        if self.verbose and not matched.all():
            unmatched_count = (~matched).sum()
            print(f"警告: 有 {unmatched_count} 个像素颜色不在 COLOR_TO_LABEL 映射中，已默认置为 0 类别。")

        return label_array

    def gray_label_to_color_image(self, label_tensor: torch.Tensor) -> np.ndarray:
        """
        输入灰度标签图（0~3），输出彩色 mask（H, W, 3）。

        Args:
            label_array: np.ndarray, dtype=uint8/int64, shape=(H, W)，值 ∈ {0,1,2,3}

        Returns:
            color_image: np.ndarray, dtype=uint8, shape=(H, W, 3)
        """
        if label_tensor.ndim == 3 and label_tensor.size(0) == 1:
            label_array = label_tensor.squeeze(0).cpu().numpy()
        elif label_tensor.ndim == 2:
            label_array = label_tensor.cpu().numpy()
        else:
            raise ValueError("label_tensor 形状必须为 (1, H, W) 或 (H, W)")


        h, w = label_array.shape
        color_image = np.zeros((h, w, 3), dtype=np.uint8)

        for label, (r, g, b) in self.label_to_color.items():
            cond = (label_array == label)
            color_image[cond] = (r, g, b)

        return color_image

    def denormalize_tensor_to_image(self, img_tensor: torch.Tensor,
                                    mean: torch.Tensor = IMG_MEAN,
                                    std: torch.Tensor = IMG_STD) -> np.ndarray:
        """
        反归一化图像，输入 (3, H, W) 或 (1, 3, H, W) 的 tensor，输出 HWC uint8。
        """
        # 保证是 (1, 3, H, W)
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)

        # 把 mean/std 移到和 img_tensor 相同的设备和 dtype
        device = img_tensor.device
        dtype = img_tensor.dtype
        mean = mean.to(device=device, dtype=dtype)
        std = std.to(device=device, dtype=dtype)

        # (1, 3, H, W) * (1, 3, 1, 1) 广播是兼容的
        img = img_tensor * std + mean
        img = torch.clamp(img, 0.0, 1.0)

        img = img[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img = (img * 255.0).round().astype(np.uint8)
        return img

    # ================= 批量预处理 =================

    def preprocess_masks_to_labels(self, root: Union[str, Path]) -> None:
        """
        对 root 目录下的 mask 进行预处理：
        - 读取 root/mask 下的所有图片
        - 按 color_to_label 转成灰度标签(0~3)
        - 保存到 root/labels 下（同名 .png）

        Args:
            root: str 或 Path，数据根目录，包含 'mask' 子目录
        """
        root = Path(root)
        mask_dir = root / "mask"
        labels_dir = root / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        mask_files = []
        for ext in exts:
            mask_files.extend(mask_dir.glob(ext))

        mask_files = sorted(mask_files)
        if self.verbose:
            print(f"在 {mask_dir} 下共找到 {len(mask_files)} 张 mask。")

        for i, mask_path in enumerate(mask_files, 1):
            try:
                img = Image.open(mask_path)
                label_array = self.rgb_mask_to_gray_label(img)

                label_img = Image.fromarray(label_array, mode="L")
                out_path = labels_dir / (mask_path.stem + ".png")
                label_img.save(out_path)

                if self.verbose and (i % 50 == 0 or i == len(mask_files)):
                    print(f"已处理 {i}/{len(mask_files)}: {mask_path.name} -> {out_path}")
            except Exception as e:
                print(f"处理 {mask_path} 失败: {e}")



if __name__ == "__main__":
    _DEFAULT_MAPPER = Visualizer(verbose=True)

    def rgb_mask_to_gray_label(mask_img: Image.Image) -> np.ndarray:
        return _DEFAULT_MAPPER.rgb_mask_to_gray_label(mask_img)


    def gray_label_to_color_image(label_tensor: torch.Tensor) -> np.ndarray:
        return _DEFAULT_MAPPER.gray_label_to_color_image(label_tensor)


    def denormlize_tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        return _DEFAULT_MAPPER.denormalize_tensor_to_image(tensor)


    def preprocess_masks_to_labels(root: Union[str, Path]) -> None:
        _DEFAULT_MAPPER.preprocess_masks_to_labels(root)
    preprocess_masks_to_labels("../datasets/train")
