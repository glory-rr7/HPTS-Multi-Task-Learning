import random
from typing import Dict

import cv2
import numpy as np
import torch


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _build_params(rng: random.Random, base_strength: float):
    s = _clamp01(base_strength)
    params = {
        "gamma": rng.uniform(0.97 - 0.10 * s, 0.93 - 0.20 * s),
        "global_gain": rng.uniform(1.03 + 0.20 * s, 1.12 + 0.45 * s),
        "threshold": rng.uniform(0.92 - 0.10 * s, 0.97 - 0.07 * s),
        "knee": rng.uniform(0.16 - 0.04 * s, 0.24 - 0.04 * s),
        "sat_start": rng.uniform(0.82 - 0.16 * s, 0.90 - 0.12 * s),
        "sat_drop": rng.uniform(0.10 + 0.10 * s, 0.20 + 0.25 * s),
        "bloom_sigma": rng.uniform(2.0 + 1.5 * s, 4.5 + 8.0 * s),
        "bloom_strength": rng.uniform(0.005 + 0.02 * s, 0.02 + 0.10 * s),
        "bloom_start": rng.uniform(0.78 - 0.12 * s, 0.90 - 0.10 * s),
        "hotspot_strength": rng.uniform(0.01 + 0.05 * s, 0.05 + 0.18 * s),
        "noise_std": rng.uniform(0.0, 0.0015 + 0.003 * s),
    }

    for key in ("threshold", "sat_start", "bloom_start"):
        params[key] = _clamp01(params[key])
    params["knee"] = max(1e-4, params["knee"])
    params["gamma"] = max(1e-3, params["gamma"])
    params["global_gain"] = max(1e-3, params["global_gain"])
    params["sat_drop"] = _clamp01(params["sat_drop"])
    params["bloom_strength"] = max(0.0, params["bloom_strength"])
    params["hotspot_strength"] = max(0.0, params["hotspot_strength"])
    params["noise_std"] = max(0.0, params["noise_std"])
    return params


def _build_hotspot_gain(h: int, w: int, rng: random.Random, strength: float):
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    cx = rng.uniform(0.2 * w, 0.8 * w)
    cy = rng.uniform(0.2 * h, 0.8 * h)
    sx = max(1.0, rng.uniform(0.22 * w, 0.65 * w))
    sy = max(1.0, rng.uniform(0.22 * h, 0.65 * h))
    gaussian = np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
    gain_map = 1.0 + strength * gaussian
    return gain_map[..., None].astype(np.float32)


def _soft_clip(values: np.ndarray, threshold: float, knee: float):
    t = float(np.clip(threshold, 0.0, 1.0))
    k = max(1e-6, float(knee))
    below = np.minimum(values, t)
    above = np.maximum(values - t, 0.0)
    compressed = below + (1.0 - np.exp(-above / k)) * (1.0 - t)
    return compressed


def _apply_overexposure(img_rgb: np.ndarray, params, rng: random.Random, np_rng: np.random.RandomState):
    h, w = img_rgb.shape[:2]
    img = img_rgb.astype(np.float32) / 255.0

    img = np.power(np.clip(img, 0.0, 1.0), params["gamma"])
    img *= params["global_gain"]
    img *= _build_hotspot_gain(h, w, rng, params["hotspot_strength"])
    img = _soft_clip(img, threshold=params["threshold"], knee=params["knee"])

    luma = np.max(img, axis=2, keepdims=True)
    sat_mask = np.clip(
        (luma - params["sat_start"]) / max(1e-6, 1.0 - params["sat_start"]),
        0.0,
        1.0,
    )
    gray = np.mean(img, axis=2, keepdims=True)
    sat_mix = sat_mask * params["sat_drop"]
    img = img * (1.0 - sat_mix) + gray * sat_mix

    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=params["bloom_sigma"], sigmaY=params["bloom_sigma"])
    bloom_mask = np.clip(
        (luma - params["bloom_start"]) / max(1e-6, 1.0 - params["bloom_start"]),
        0.0,
        1.0,
    )
    img = img + blur * params["bloom_strength"] * bloom_mask

    if params["noise_std"] > 0:
        noise = np_rng.normal(0.0, params["noise_std"], size=img.shape).astype(np.float32)
        img = img + noise

    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def apply_overexposure_image(
    data: Dict[str, torch.Tensor],
    strength: float = 0.7,
    strength_jitter: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """
    过曝光增强，仅修改 image，mask/rebuild 原样返回。
    """
    image = data["image"]
    rng = random
    np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))

    sample_strength = _clamp01(strength + rng.uniform(-strength_jitter, strength_jitter))
    params = _build_params(rng, sample_strength)

    image_np = image.detach().cpu().permute(1, 2, 0).contiguous().numpy()
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    aug_image_np = _apply_overexposure(image_np, params, rng=rng, np_rng=np_rng)

    output = dict(data)
    output["image"] = (
        torch.from_numpy(np.ascontiguousarray(aug_image_np))
        .permute(2, 0, 1)
        .to(device=image.device, dtype=image.dtype)
    )
    return output
