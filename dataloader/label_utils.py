import os

import torch


LEGACY_RGB_TO_CLASS = {
    (255, 255, 255): 0,
    (0, 255, 0): 1,
    (255, 0, 0): 2,
    (0, 0, 255): 3,
}
SIGNATR_RGB_TO_CLASS = {
    (0, 0, 255): 0,
    (0, 255, 0): 1,
    (255, 0, 0): 2,
    (255, 255, 0): 3,
}


def resolve_dataset_directory(root_path, *candidate_names):
    """Return the first existing directory from supported dataset layouts."""
    for name in candidate_names:
        path = os.path.join(root_path, name)
        if os.path.isdir(path):
            return path
    expected = ", ".join(os.path.join(root_path, name) for name in candidate_names)
    raise FileNotFoundError(f"None of the supported dataset directories exists: {expected}")


def decode_segmentation_label(label, label_path="<tensor>"):
    """Decode a grayscale class map or an RGB color mask to shape (1,H,W)."""
    if label.ndim != 3:
        raise ValueError(f"Label must have shape (C,H,W), got {tuple(label.shape)}: {label_path}")
    if label.shape[0] == 1:
        decoded = label.long()
    elif label.shape[0] >= 3:
        rgb = label[:3]
        unique_colors = {
            tuple(int(channel) for channel in color)
            for color in torch.unique(rgb.permute(1, 2, 0).reshape(-1, 3), dim=0).cpu().tolist()
        }
        has_white = (255, 255, 255) in unique_colors
        has_blue = (0, 0, 255) in unique_colors
        has_yellow = (255, 255, 0) in unique_colors
        if has_white and has_yellow:
            raise ValueError(f"Label mixes legacy and SignaTR RGB palettes: {label_path}")
        if has_yellow or (has_blue and not has_white):
            palette = SIGNATR_RGB_TO_CLASS
        elif has_white:
            palette = LEGACY_RGB_TO_CLASS
        else:
            raise ValueError(
                "Cannot identify RGB label palette; expected white background for the legacy "
                f"palette or yellow overlap for the SignaTR palette: {label_path}"
            )
        decoded = torch.full((1, rgb.shape[1], rgb.shape[2]), 255,
                             dtype=torch.long, device=rgb.device)
        for color, class_id in palette.items():
            color_tensor = torch.tensor(color, dtype=rgb.dtype, device=rgb.device).view(3, 1, 1)
            decoded[0, torch.all(rgb == color_tensor, dim=0)] = class_id
    else:
        raise ValueError(f"Unsupported label channel count {label.shape[0]}: {label_path}")

    invalid = (decoded < 0) | (decoded > 3)
    if invalid.any():
        invalid_count = int(invalid.sum().item())
        raise ValueError(
            f"Label contains {invalid_count} unrecognized pixels; expected class IDs 0-3 "
            f"or one supported exact RGB palette: {label_path}"
        )
    return decoded
