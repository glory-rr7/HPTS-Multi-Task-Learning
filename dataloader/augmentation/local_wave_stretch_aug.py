import random
from typing import Dict, Tuple

import cv2
import numpy as np
import torch


def _to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return image_np


def _mask_to_hw(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().squeeze(0).contiguous().numpy()
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)
    return mask_np


def _to_chw_tensor(image_np: np.ndarray, ref_tensor: torch.Tensor) -> torch.Tensor:
    out = torch.from_numpy(np.ascontiguousarray(image_np)).permute(2, 0, 1)
    return out.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


def _to_mask_tensor(mask_np: np.ndarray, ref_tensor: torch.Tensor) -> torch.Tensor:
    out = torch.from_numpy(np.ascontiguousarray(mask_np)).unsqueeze(0)
    return out.to(device=ref_tensor.device, dtype=ref_tensor.dtype)


def local_wave_stretch_map(
    h: int,
    w: int,
    rng: random.Random,
    num_waves: int = 5,
    amp_y_range: Tuple[float, float] = (3.0, 12.0),
    amp_x_range: Tuple[float, float] = (0.4, 2.5),
    wavelength_range: Tuple[float, float] = (48.0, 220.0),
    window_ratio_range: Tuple[float, float] = (0.10, 0.36),
):
    """
    模拟扫描走纸拉伸：
    在多个局部高斯窗口叠加不同方向/频率/相位的波动。
    """
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    min_side = float(min(h, w))
    wave_min = max(18.0, min_side * 0.05, float(wavelength_range[0]))
    wave_max = max(wave_min + 1.0, float(wavelength_range[1]))

    for _ in range(max(1, num_waves)):
        cx = rng.uniform(0.0, w - 1.0)
        cy = rng.uniform(0.0, h - 1.0)
        sx = max(8.0, rng.uniform(*window_ratio_range) * w)
        sy = max(8.0, rng.uniform(*window_ratio_range) * h)

        theta = rng.uniform(-np.pi, np.pi)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        wavelength = rng.uniform(wave_min, wave_max)
        amp_y = rng.uniform(*amp_y_range) * rng.choice((-1.0, 1.0))
        amp_x = rng.uniform(*amp_x_range) * rng.choice((-1.0, 1.0))

        projected = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
        base_wave = np.sin((2.0 * np.pi / wavelength) * projected + phase)
        harmonic = 0.20 * np.sin((4.0 * np.pi / wavelength) * projected + 1.3 * phase)
        wave = base_wave + harmonic

        window = np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
        map_y += (amp_y * wave * window).astype(np.float32)
        map_x += (amp_x * wave * window).astype(np.float32)

    # 叠加一层低频行波
    row_period = rng.uniform(40.0, 130.0)
    row_phase = rng.uniform(0.0, 2.0 * np.pi)
    row_amp = rng.uniform(0.4, 1.4)
    row_wave = np.sin(2.0 * np.pi * np.arange(h, dtype=np.float32) / row_period + row_phase)[:, None]
    map_x += row_wave * row_amp

    return map_x, map_y


def _apply_remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, interp: int) -> np.ndarray:
    h, w = image.shape[:2]
    remap_x = np.clip(map_x, 0, w - 1)
    remap_y = np.clip(map_y, 0, h - 1)
    return cv2.remap(
        image,
        remap_x,
        remap_y,
        interpolation=interp,
        borderMode=cv2.BORDER_REPLICATE,
    )


def apply_local_wave_stretch(
    data: Dict[str, torch.Tensor],
    num_waves_range=(2, 4),
    amp_y_range=(7.0, 11.0),
    amp_x_range=(1.2, 4.0),
    window_ratio_range=(0.12, 0.70),
) -> Dict[str, torch.Tensor]:
    """
    扫描局部波动拉伸增强，同步作用于 image/mask/rebuild。
    """
    image = data["image"]
    mask = data["mask"]
    rebuild = data["rebuild"]

    h, w = image.shape[-2], image.shape[-1]
    rng = random

    wavelength_range = (
        max(24.0, 0.05 * min(h, w)),
        max(140.0, 0.30 * max(h, w)),
    )
    num_waves = rng.randint(num_waves_range[0], num_waves_range[1])
    map_x, map_y = local_wave_stretch_map(
        h=h,
        w=w,
        rng=rng,
        num_waves=num_waves,
        amp_y_range=amp_y_range,
        amp_x_range=amp_x_range,
        wavelength_range=wavelength_range,
        window_ratio_range=window_ratio_range,
    )

    image_np = _to_hwc_uint8(image)
    rebuild_np = _to_hwc_uint8(rebuild)
    mask_np = _mask_to_hw(mask)

    aug_image_np = _apply_remap(image_np, map_x, map_y, interp=cv2.INTER_LINEAR)
    aug_rebuild_np = _apply_remap(rebuild_np, map_x, map_y, interp=cv2.INTER_LINEAR)
    aug_mask_np = _apply_remap(mask_np, map_x, map_y, interp=cv2.INTER_NEAREST)

    output = dict(data)
    output["image"] = _to_chw_tensor(aug_image_np, image)
    output["rebuild"] = _to_chw_tensor(aug_rebuild_np, rebuild)
    output["mask"] = _to_mask_tensor(aug_mask_np, mask)
    return output
