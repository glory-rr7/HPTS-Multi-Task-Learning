import random
from typing import Dict, List, Callable

import torch

from .bond_color_shift_aug import apply_bond_color_shift
from .edge_artifact_image_aug import apply_edge_artifact_image
from .local_wave_stretch_aug import apply_local_wave_stretch
from .overexposure_image_aug import apply_overexposure_image
from .scaning_local_jitter_aug import apply_scaning_local_jitter

AugFn = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]

AUGMENTATION_METHODS: List[AugFn] = [
    apply_scaning_local_jitter,
    apply_local_wave_stretch,
    apply_overexposure_image,
    apply_edge_artifact_image,
    apply_bond_color_shift,
]


def apply_random_augmentation(
    data: Dict[str, torch.Tensor],
    enabled: bool = False,
    apply_prob: float = 0.6,
) -> Dict[str, torch.Tensor]:
    """
    统一增强调度入口：
      - enabled=False: 直接返回
      - enabled=True: 以 apply_prob 触发增强
      - 触发后在增强池中等概率随机选择一种
    """
    if not enabled:
        return data

    if random.random() >= apply_prob:
        return data

    aug_fn = random.choice(AUGMENTATION_METHODS)
    return aug_fn(data)


__all__ = [
    "apply_scaning_local_jitter",
    "apply_local_wave_stretch",
    "apply_overexposure_image",
    "apply_edge_artifact_image",
    "apply_bond_color_shift",
    "apply_random_augmentation",
    "AUGMENTATION_METHODS",
]
