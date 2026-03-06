import random
from typing import Dict

import torch


def apply_bond_color_shift(
    data: Dict[str, torch.Tensor],
    shift_intensity: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    颜色偏移增强，按 image/rebuild 的不同语义分别处理。

    mask 为单通道标签图 (1,H,W)，类别取值:
      0: background
      1: printed
      2: handwriting
      3: overlap

    image:
      - printed(1) 使用 printed 偏移
      - handwriting(2) 使用 handwriting 偏移
      - overlap(3) 随机归入 printed 或 handwriting

    rebuild:
      - 仅保留 background/printed 语义
      - printed(1) 与 overlap(3) 都使用 printed 偏移
      - handwriting(2) 视为非打印区域，不施加手写偏移
    """
    image = data["image"]
    rebuild = data["rebuild"]
    mask = data["mask"]

    image_float = image.float() / 255.0
    rebuild_float = rebuild.float() / 255.0
    label_map = mask.long().squeeze(0)

    assign_overlap_to_printed = random.random() > 0.5
    printed_mask = label_map == 1
    handwriting_mask = label_map == 2
    overlap_mask = label_map == 3

    if assign_overlap_to_printed:
        image_printed_mask = printed_mask | overlap_mask
        image_handwriting_mask = handwriting_mask
    else:
        image_printed_mask = printed_mask
        image_handwriting_mask = handwriting_mask | overlap_mask

    rebuild_printed_mask = printed_mask | overlap_mask

    c = image.shape[0]
    printed_delta = (torch.rand(c, 1, 1, device=image.device, dtype=torch.float32) * 2.0 - 1.0) * shift_intensity
    handwriting_delta = (torch.rand(c, 1, 1, device=image.device, dtype=torch.float32) * 2.0 - 1.0) * shift_intensity

    image_printed_mask = image_printed_mask.unsqueeze(0).expand(c, -1, -1).float().to(image.device)
    image_handwriting_mask = image_handwriting_mask.unsqueeze(0).expand(c, -1, -1).float().to(image.device)
    rebuild_printed_mask = rebuild_printed_mask.unsqueeze(0).expand(c, -1, -1).float().to(image.device)

    image_float = (
        image_float
        + printed_delta * image_printed_mask
        + handwriting_delta * image_handwriting_mask
    )
    rebuild_float = rebuild_float + printed_delta * rebuild_printed_mask

    image_out = torch.clamp(torch.round(torch.clamp(image_float, 0.0, 1.0) * 255.0), 0, 255).to(dtype=image.dtype)
    rebuild_out = torch.clamp(torch.round(torch.clamp(rebuild_float, 0.0, 1.0) * 255.0), 0, 255).to(dtype=rebuild.dtype)

    output = dict(data)
    output["image"] = image_out
    output["rebuild"] = rebuild_out
    return output
