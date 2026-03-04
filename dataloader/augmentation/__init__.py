import random
from typing import Dict, List, Callable, Sequence

import torch

from .bond_color_shift_aug import apply_bond_color_shift
from .edge_artifact_image_aug import apply_edge_artifact_image
from .local_wave_stretch_aug import apply_local_wave_stretch
from .overexposure_image_aug import apply_overexposure_image
from .scaning_local_jitter_aug import apply_scaning_local_jitter

AugFn = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]

# 分组增强配置
AUG_GROUP_A_METHODS: List[AugFn] = [
    apply_bond_color_shift,
    apply_overexposure_image,
    apply_local_wave_stretch,
]

AUG_GROUP_B_METHODS: List[AugFn] = [
    apply_edge_artifact_image,
]

# 兼容旧导出：保留原增强池定义（不再作为随机调度来源）
AUGMENTATION_METHODS: List[AugFn] = [
    apply_scaning_local_jitter,
    apply_local_wave_stretch,
    apply_overexposure_image,
    apply_edge_artifact_image,
    apply_bond_color_shift,
]

AUGMENTATION_GROUP_CHOICES = ("A", "B", "A+B")


def _apply_aug_sequence(
    data: Dict[str, torch.Tensor],
    aug_methods: Sequence[AugFn],
) -> Dict[str, torch.Tensor]:
    output = data
    for aug_fn in aug_methods:
        output = aug_fn(output)
    return output


def _sample_group_methods(
    aug_methods: Sequence[AugFn],
    method_prob: float = 0.5,
) -> List[AugFn]:
    if not aug_methods:
        return []

    selected = [fn for fn in aug_methods if random.random() < method_prob]
    # 保底至少执行一个增强，避免触发增强后完全无变化
    if not selected:
        selected = [random.choice(list(aug_methods))]
    return selected


def apply_group_a_augmentation(
    data: Dict[str, torch.Tensor],
    method_prob: float = 0.5,
) -> Dict[str, torch.Tensor]:
    methods = _sample_group_methods(AUG_GROUP_A_METHODS, method_prob=method_prob)
    return _apply_aug_sequence(data, methods)


def apply_group_b_augmentation(
    data: Dict[str, torch.Tensor],
    method_prob: float = 0.5,
) -> Dict[str, torch.Tensor]:
    methods = _sample_group_methods(AUG_GROUP_B_METHODS, method_prob=method_prob)
    return _apply_aug_sequence(data, methods)


def apply_group_ab_augmentation(
    data: Dict[str, torch.Tensor],
    method_prob: float = 0.5,
) -> Dict[str, torch.Tensor]:
    output = apply_group_a_augmentation(data, method_prob=method_prob)
    output = apply_group_b_augmentation(output, method_prob=method_prob)
    return output


def apply_random_augmentation(
    data: Dict[str, torch.Tensor],
    enabled: bool = False,
    apply_prob: float = 0.75,
    method_prob: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """
    统一增强调度入口：
      - enabled=False: 直接返回
      - enabled=True: 以 apply_prob 触发增强
      - 触发后从 A / B / A+B 三种策略中等概率采样
          A: 边缘伪影
          B: 在 [过曝光, 颜色偏移, 局部波动拉伸] 中随机抽样后顺序叠加
          A+B: 先 A 再 B
      - 组内方法按 method_prob 随机抽样，且每组至少保底 1 个方法生效
    """
    if not enabled:
        return data

    if random.random() >= apply_prob:
        return data

    choice = random.choice(AUGMENTATION_GROUP_CHOICES)
    if choice == "A":
        return apply_group_a_augmentation(data, method_prob=method_prob)
    if choice == "B":
        return apply_group_b_augmentation(data, method_prob=method_prob)
    return apply_group_ab_augmentation(data, method_prob=method_prob)


__all__ = [
    "apply_scaning_local_jitter",
    "apply_local_wave_stretch",
    "apply_overexposure_image",
    "apply_edge_artifact_image",
    "apply_bond_color_shift",
    "apply_group_a_augmentation",
    "apply_group_b_augmentation",
    "apply_group_ab_augmentation",
    "apply_random_augmentation",
    "AUG_GROUP_A_METHODS",
    "AUG_GROUP_B_METHODS",
    "AUGMENTATION_GROUP_CHOICES",
    "AUGMENTATION_METHODS",
]
