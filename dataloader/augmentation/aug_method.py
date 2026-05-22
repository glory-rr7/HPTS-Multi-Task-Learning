import torch
import random
from typing import Tuple, Union
import torchvision.io as tvio
from pathlib import Path
import torch.nn.functional as F

def edge_artifact(
        img: torch.Tensor,
        side: str = "right",  # "top", "bottom", "left", "right"
        width_ratio: float = 0.4,
        mode: str = "lighten",  # "lighten" 或 "darken"
        strength: float = 0.5
) -> torch.Tensor:
    """
    要求:
      - 输入 img 为 float 张量，值域约在 [0, 1]
      - 支持 (C,H,W) 或 (B,C,H,W)
    """
    assert img.dim() in (3, 4), f"img.dim must be 3 or 4, got {img.dim()}"

    squeeze_batch = False
    if img.dim() == 3:
        # (C,H,W) -> (1,C,H,W)
        img = img.unsqueeze(0)
        squeeze_batch = True

    B, C, H, W = img.shape
    out = img.clone()

    # 计算边缘宽度
    if side in ["top", "bottom"]:
        edge_w = int(H * width_ratio)
    else:
        edge_w = int(W * width_ratio)

    if edge_w <= 0:
        return img.squeeze(0) if squeeze_batch else img

    # 取出边缘区域
    if side == "top":
        slc = (slice(None), slice(None), slice(0, edge_w), slice(None))
    elif side == "bottom":
        slc = (slice(None), slice(None), slice(H - edge_w, H), slice(None))
    elif side == "left":
        slc = (slice(None), slice(None), slice(None), slice(0, edge_w))
    elif side == "right":
        slc = (slice(None), slice(None), slice(None), slice(W - edge_w, W))
    else:
        return img.squeeze(0) if squeeze_batch else img

    region = out[slc]

    if mode == "lighten":
        # 往 1.0 拉近：region_new = region*(1-strength) + strength
        region_new = region + strength * (1.0 - region)
    else:  # "darken"
        # 往 0.0 压：region_new = region*(1-strength)
        region_new = region * (1.0 - strength)

    region_new = torch.clamp(region_new, 0.0, 1.0)
    out[slc] = region_new

    if squeeze_batch:
        out = out.squeeze(0)
    return out


def call_edge_artifact(data: dict) -> dict:
    """
    随机调用 edge_artifact，对data字典中的'image'进行增强
      - side: 上/下/左/右 等概率
      - width_ratio: 0.3 ~ 0.7
      - mode: lighten / darken 各一半
      - strength: 0.3 ~ 0.7

    Args:
        data: 字典，必须包含 'image' 键，其值为 tensor (C,H,W) 或 (B,C,H,W)，值域 [0,1]

    Returns:
        增强后的data字典（会修改'image'字段）
    """
    if 'image' not in data:
        return data

    img = data['image']

    # 随机参数
    side = random.choice(["top", "bottom", "left", "right"])
    width_ratio = random.uniform(0.3, 0.7)
    mode = "lighten" if random.random() < 0.5 else "darken"
    strength = random.uniform(0.3, 0.7)

    # 应用边缘伪影
    augmented_img = edge_artifact(
        img,
        side=side,
        width_ratio=width_ratio,
        mode=mode,
        strength=strength,
    )

    # 更新字典
    data['image'] = augmented_img

    return data


def gaussian_blur(
    img: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    高斯模糊
    要求:
      - img: float, ∈[0,1], (C,H,W) 或 (B,C,H,W)
    """
    assert img.dim() in (3, 4), f"img.dim must be 3 or 4, got {img.dim()}"
    assert kernel_size % 2 == 1, "kernel_size must be odd"

    squeeze_batch = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze_batch = True

    B, C, H, W = img.shape
    if H < 2 or W < 2:
        return img.squeeze(0) if squeeze_batch else img

    # 构造二维高斯核
    k = kernel_size
    coords = torch.arange(k, device=img.device) - (k - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    gauss2d = torch.outer(g, g)
    gauss2d = gauss2d / gauss2d.sum()

    # depthwise conv
    weight = gauss2d.view(1, 1, k, k).repeat(C, 1, 1, 1)  # (C,1,k,k)

    pad = k // 2
    x = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    out = F.conv2d(x, weight, bias=None, stride=1, padding=0, groups=C)
    out = torch.clamp(out, 0.0, 1.0)

    if squeeze_batch:
        out = out.squeeze(0)
    return out


def call_gaussian_blur(img: torch.Tensor) -> torch.Tensor:
    """
    随机高斯模糊:
      - kernel_size: 3 / 5 / 7
      - sigma: 0.5 ~ 2.0
    """
    kernel_size = random.choice([3, 5, 7])
    sigma = random.uniform(0.5, 2.0)
    return gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)


def color_jitter_simple(
    img: torch.Tensor,
    brightness: float = 0.0,
    contrast: float = 0.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    简单颜色扰动:
      - brightness: [-0.5, 0.5] 加性偏移
      - contrast:  [0.5, 1.5] 乘性对比度
      - gamma:     [0.7, 1.3] gamma 矫正
    要求 img ∈[0,1], (C,H,W) 或 (B,C,H,W)
    """
    assert img.dim() in (3, 4), f"img.dim must be 3 or 4, got {img.dim()}"

    squeeze_batch = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze_batch = True

    x = img

    # 亮度: 加偏移
    if brightness != 0.0:
        x = x + brightness

    # 对比度: 以每张图的均值为中心
    if contrast != 0.0:
        mean = x.mean(dim=(2, 3), keepdim=True)
        x = (x - mean) * contrast + mean

    x = torch.clamp(x, 0.0, 1.0)

    # gamma
    if abs(gamma - 1.0) > 1e-3:
        # 避免 0 ** gamma 的数值问题，先 clamp
        x = torch.clamp(x, 1e-6, 1.0) ** gamma

    x = torch.clamp(x, 0.0, 1.0)

    if squeeze_batch:
        x = x.squeeze(0)
    return x


def call_color_jitter(img: torch.Tensor) -> torch.Tensor:
    """
    随机颜色扰动:
      - brightness: -0.2 ~ 0.2
      - contrast:   0.8 ~ 1.2
      - gamma:      0.8 ~ 1.2
    """
    brightness = random.uniform(-0.2, 0.2)
    contrast = random.uniform(0.8, 1.2)
    gamma = random.uniform(0.8, 1.2)
    return color_jitter_simple(
        img,
        brightness=brightness,
        contrast=contrast,
        gamma=gamma,
    )
def add_gaussian_noise(
    img: torch.Tensor,
    sigma: float = 0.05,
) -> torch.Tensor:
    """
    高斯噪声: x' = x + N(0, sigma^2)
    要求 img ∈[0,1], (C,H,W) 或 (B,C,H,W)
    """
    assert img.dim() in (3, 4), f"img.dim must be 3 or 4, got {img.dim()}"

    noise = torch.randn_like(img) * sigma
    out = img + noise
    out = torch.clamp(out, 0.0, 1.0)
    return out


def call_gaussian_noise(img: torch.Tensor) -> torch.Tensor:
    """
    随机高斯噪声:
      - sigma: 0.01 ~ 0.1
    """
    sigma = random.uniform(0.01, 0.1)
    return add_gaussian_noise(img, sigma=sigma)


def elastic_distortion(
    img: torch.Tensor,
    alpha: float = 5.0,
    sigma: float = 8.0,
) -> torch.Tensor:
    """
    弹性扭曲 (elastic distortion)
    - alpha: 位移强度
    - sigma: 高斯平滑强度
    要求 img ∈[0,1], (C,H,W) 或 (B,C,H,W)
    """
    assert img.dim() in (3, 4), f"img.dim must be 3 or 4, got {img.dim()}"

    squeeze_batch = False
    if img.dim() == 3:
        img = img.unsqueeze(0)
        squeeze_batch = True

    B, C, H, W = img.shape
    if H < 2 or W < 2:
        return img.squeeze(0) if squeeze_batch else img

    # 生成基础网格 [-1,1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=img.device),
        torch.linspace(-1, 1, W, device=img.device),
        indexing="ij",
    )
    base_grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B,H,W,2)

    # 生成随机位移场 (B, 2, H, W)
    disp = torch.randn(B, 2, H, W, device=img.device)

    # 用高斯模糊平滑位移场
    k = int(4 * sigma) | 1  # 近似 kernel_size，保证奇数
    if k > 1:
        coords = torch.arange(k, device=img.device) - (k - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        gauss2d = torch.outer(g, g)
        gauss2d = gauss2d / gauss2d.sum()

        weight = gauss2d.view(1, 1, k, k)
        pad = k // 2
        disp_x = disp[:, 0:1]
        disp_y = disp[:, 1:2]
        disp_x = F.pad(disp_x, (pad, pad, pad, pad), mode="reflect")
        disp_y = F.pad(disp_y, (pad, pad, pad, pad), mode="reflect")
        disp_x = F.conv2d(disp_x, weight, padding=0)
        disp_y = F.conv2d(disp_y, weight, padding=0)
        disp = torch.cat([disp_x, disp_y], dim=1)

    # 归一化位移到 [-1,1] 坐标尺度
    disp = disp.permute(0, 2, 3, 1)  # (B,H,W,2)
    disp = disp * (alpha / max(H, W))

    grid = base_grid + disp
    grid = torch.clamp(grid, -1.0, 1.0)

    out = F.grid_sample(
        img,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    out = torch.clamp(out, 0.0, 1.0)

    if squeeze_batch:
        out = out.squeeze(0)
    return out


def call_elastic_distortion(img: torch.Tensor) -> torch.Tensor:
    """
    随机弹性扭曲:
      - alpha: 2.0 ~ 10.0
      - sigma: 4.0 ~ 10.0
    """
    alpha = random.uniform(2.0, 10.0)
    sigma = random.uniform(4.0, 10.0)
    return elastic_distortion(img, alpha=alpha, sigma=sigma)



# 必须使用你的数据集定义的 Mean 和 Std；当前默认是 raw [0,1] 空间
NORM_MEAN = [0.0, 0.0, 0.0]
NORM_STD = [1.0, 1.0, 1.0]


def _bond_color_shifted(data, mean=NORM_MEAN, std=NORM_STD):
    """
    颜色偏移增强 (支持外部调用，自动按当前 mean/std 做反归一化和再归一化)
    :param data: 字典 {'image': tensor, 'label': tensor, 'rebuild': tensor}
    :param mean: 归一化均值列表
    :param std: 归一化标准差列表
    :return: 经过颜色偏移和再归一化的 data
    """
    image = data['image']
    label = data['label']
    rebuild = data['rebuild']

    # 1. 维度检查与适配 (处理 [C, H, W] 或 [B, C, H, W] 的情况)
    has_batch = image.ndim == 4
    if not has_batch:
        image = image.unsqueeze(0)
        label = label.unsqueeze(0)
        rebuild = rebuild.unsqueeze(0)

    b, c, h, w = image.shape
    device = image.device
    dtype = image.dtype

    # 2. 逆归一化 (De-normalization)
    # 将归一化数据变回 [0, 1] 范围
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, c, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, c, 1, 1)

    # I_unnormalized = I_normalized * std + mean
    image = image * std_tensor + mean_tensor
    rebuild = rebuild * std_tensor + mean_tensor

    # 在进行颜色偏移之前，必须先将数据截断到 [0, 1]，防止逆归一化导致超出范围
    image = torch.clamp(image, 0.0, 1.0)
    rebuild = torch.clamp(rebuild, 0.0, 1.0)

    # 3. 核心颜色偏移逻辑 (在 [0, 1] 范围内操作)

    # 准备 Mask
    if label.ndim == 3: label = label.unsqueeze(1)
    label_map = label[:, 0:1, :, :]

    assign_3_to_1 = random.random() > 0.5
    mask_1_base = (label_map == 1)
    mask_2_base = (label_map == 2)
    mask_3_base = (label_map == 3)

    if assign_3_to_1:
        final_mask1 = mask_1_base | mask_3_base
        final_mask2 = mask_2_base
    else:
        final_mask1 = mask_1_base
        final_mask2 = mask_2_base | mask_3_base

    # 生成颜色偏移
    shift_intensity = 0.5

    def get_shift(bs, ch):
        shift = torch.rand(bs, ch, 1, 1, device=device, dtype=dtype) * 2 - 1
        return shift * shift_intensity

    delta_1 = get_shift(b, c)
    delta_2 = get_shift(b, c)

    final_mask1 = final_mask1.expand_as(image)
    final_mask2 = final_mask2.expand_as(image)

    # 应用偏移
    image = image + delta_1 * final_mask1.float() + delta_2 * final_mask2.float()
    rebuild = rebuild + delta_1 * final_mask1.float() + delta_2 * final_mask2.float()

    # 必须在再归一化前截断 (Clamp)，确保没有超过 1.0
    image = torch.clamp(image, 0.0, 1.0)
    rebuild = torch.clamp(rebuild, 0.0, 1.0)

    # 4. 再归一化 (Re-normalization)
    # I_normalized = (I_unnormalized - mean) / std
    image = (image - mean_tensor) / std_tensor
    rebuild = (rebuild - mean_tensor) / std_tensor

    # 5. 还原维度
    if not has_batch:
        image = image.squeeze(0)
        label = label.squeeze(0)
        rebuild = rebuild.squeeze(0)

    data['image'] = image
    data['label'] = label
    data['rebuild'] = rebuild
    return data


def call_bond_color_shifted(data):
    """
    外部调用的包装函数，如果需要传递自定义的 mean/std，请修改此函数或 _bond_color_shifted 的默认参数。
    """
    return _bond_color_shifted(data)
