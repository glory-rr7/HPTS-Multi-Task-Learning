import random
from typing import Dict

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


def _random_region_jitter_map(
    h: int,
    w: int,
    rng: random.Random,
    np_rng: np.random.RandomState,
    num_regions: int = 3,
    area_ratio=(0.08, 0.25),
    max_amplitude: float = 4.0,
    sigma: float = 8.0,
    field_num_range=(2, 5),
):
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    for _ in range(num_regions):
        ratio = rng.uniform(*area_ratio)
        region_h = max(1, int(h * ratio))
        region_w = max(1, int(w * ratio))
        region_h = min(region_h, h)
        region_w = min(region_w, w)

        y1 = rng.randint(0, h - region_h)
        x1 = rng.randint(0, w - region_w)
        y2 = y1 + region_h
        x2 = x1 + region_w

        rh, rw = region_h, region_w
        dx_total = np.zeros((rh, rw), np.float32)
        dy_total = np.zeros((rh, rw), np.float32)

        field_num = rng.randint(field_num_range[0], field_num_range[1])
        for _ in range(field_num):
            dx = np_rng.rand(rh, rw).astype(np.float32) * 2.0 - 1.0
            dy = np_rng.rand(rh, rw).astype(np.float32) * 2.0 - 1.0

            dx = cv2.GaussianBlur(dx, (0, 0), sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)

            amp = rng.uniform(1.2, max_amplitude)
            dx_total += dx * amp
            dy_total += dy * amp

        x, y = np.meshgrid(np.arange(rw), np.arange(rh))
        map_x[y1:y2, x1:x2] = (x + dx_total + x1).astype(np.float32)
        map_y[y1:y2, x1:x2] = (y + dy_total + y1).astype(np.float32)

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


def apply_scaning_local_jitter(
    data: Dict[str, torch.Tensor],
    num_regions_range=(2, 5),
    area_ratio=(0.08, 0.25),
    max_amplitude_range=(2.5, 6.0),
    sigma_range=(6.0, 10.0),
    field_num_range=(2, 5),
) -> Dict[str, torch.Tensor]:
    """
    局部抖动增强，统一输入输出:
      data 至少包含 image/mask/rebuild
      image/rebuild: (3,H,W), mask: (1,H,W)
    """
    image = data["image"]
    mask = data["mask"]
    rebuild = data["rebuild"]

    h, w = image.shape[-2], image.shape[-1]
    rng = random
    np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

    num_regions = rng.randint(num_regions_range[0], num_regions_range[1])
    max_amplitude = rng.uniform(max_amplitude_range[0], max_amplitude_range[1])
    sigma = rng.uniform(sigma_range[0], sigma_range[1])

    map_x, map_y = _random_region_jitter_map(
        h=h,
        w=w,
        rng=rng,
        np_rng=np_rng,
        num_regions=num_regions,
        area_ratio=area_ratio,
        max_amplitude=max_amplitude,
        sigma=sigma,
        field_num_range=field_num_range,
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
