import random
from typing import Dict

import torch


def apply_bond_color_shift(
    data: Dict[str, torch.Tensor],
    shift_intensity: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    颜色偏移增强，修改 image/rebuild，mask 原样返回。
    mask 要求为单通道标签图 (1,H,W)，类别取值 0~3。
    """
    image = data["image"]
    rebuild = data["rebuild"]
    mask = data["mask"]

    image_float = image.float() / 255.0
    rebuild_float = rebuild.float() / 255.0
    label_map = mask.long().squeeze(0)

    assign_3_to_1 = random.random() > 0.5
    mask_1_base = label_map == 1
    mask_2_base = label_map == 2
    mask_3_base = label_map == 3

    if assign_3_to_1:
        final_mask1 = mask_1_base | mask_3_base
        final_mask2 = mask_2_base
    else:
        final_mask1 = mask_1_base
        final_mask2 = mask_2_base | mask_3_base

    c = image.shape[0]
    delta_1 = (torch.rand(c, 1, 1, device=image.device, dtype=torch.float32) * 2.0 - 1.0) * shift_intensity
    delta_2 = (torch.rand(c, 1, 1, device=image.device, dtype=torch.float32) * 2.0 - 1.0) * shift_intensity

    final_mask1 = final_mask1.unsqueeze(0).expand(c, -1, -1).float().to(image.device)
    final_mask2 = final_mask2.unsqueeze(0).expand(c, -1, -1).float().to(image.device)

    image_float = image_float + delta_1 * final_mask1 + delta_2 * final_mask2
    rebuild_float = rebuild_float + delta_1 * final_mask1 + delta_2 * final_mask2

    image_out = torch.clamp(torch.round(torch.clamp(image_float, 0.0, 1.0) * 255.0), 0, 255).to(dtype=image.dtype)
    rebuild_out = torch.clamp(torch.round(torch.clamp(rebuild_float, 0.0, 1.0) * 255.0), 0, 255).to(dtype=rebuild.dtype)

    output = dict(data)
    output["image"] = image_out
    output["rebuild"] = rebuild_out
    return output
