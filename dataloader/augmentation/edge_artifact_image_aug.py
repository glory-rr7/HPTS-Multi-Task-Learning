import random
from typing import Dict

import torch

from .aug_method import edge_artifact


def apply_edge_artifact_image(
    data: Dict[str, torch.Tensor],
    width_ratio_range=(0.3, 0.7),
    strength_range=(0.3, 0.7),
) -> Dict[str, torch.Tensor]:
    """
    边缘伪影增强，仅修改 image，mask/rebuild 原样返回。
    """
    image = data["image"]
    image_float = image.float() / 255.0

    side = random.choice(["top", "bottom", "left", "right"])
    width_ratio = random.uniform(width_ratio_range[0], width_ratio_range[1])
    mode = "lighten" if random.random() < 0.5 else "darken"
    strength = random.uniform(strength_range[0], strength_range[1])

    aug_image_float = edge_artifact(
        image_float,
        side=side,
        width_ratio=width_ratio,
        mode=mode,
        strength=strength,
    )
    aug_image = torch.clamp(torch.round(aug_image_float * 255.0), 0, 255).to(dtype=image.dtype)

    output = dict(data)
    output["image"] = aug_image.to(device=image.device)
    return output
