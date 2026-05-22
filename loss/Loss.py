import torch
from torch import nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import numpy as np


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram
def visual(image):
    im = image.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

def dice_loss(input, target):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    input = input
    target = target

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

#用来评估模型的性能，不用于训练模型

def SimilarityLoss(
    fake,
    gt,
    *,
    mean=(0.0, 0.0, 0.0),
    std=(1.0, 1.0, 1.0),
    SSIM=True,
):
    """
    fake / gt : 4-D tensor  [B, C, H, W]。
    按给定 mean/std 反标准化回 0-1 后计算 L1 / MSE / PSNR / SSIM。
    """

    # ---- 1. 反标准化：x = x * std + mean  ---------------------------------
    mean = torch.as_tensor(mean, dtype=fake.dtype, device=fake.device)[None, :, None, None]
    std  = torch.as_tensor(std,  dtype=fake.dtype, device=fake.device)[None, :, None, None]

    fake = (fake * std + mean).clamp(0.0, 1.0)
    gt   = (gt   * std + mean).clamp(0.0, 1.0)

    # ---- 2. L1 & MSE -------------------------------------------------------
    l1_loss  = F.l1_loss(fake, gt)
    mse_loss = F.mse_loss(fake, gt)

    # ---- 3. PSNR -----------------------------------------------------------
    eps  = 1e-8                     # 防止 log(0)
    psnr = 10 * torch.log10(1.0 / (mse_loss + eps))

    if not SSIM:
        return l1_loss, mse_loss, psnr

    # ---- 4. SSIM（转 CPU / NumPy）------------------------------------------
    fake_np = fake.squeeze(0).detach().cpu().numpy()
    gt_np   = gt.squeeze(0).detach().cpu().numpy()

    if fake_np.ndim == 3:                       # (C, H, W) → (H, W, C)
        fake_np = np.transpose(fake_np, (1, 2, 0))
        gt_np   = np.transpose(gt_np,   (1, 2, 0))

    ssim_val = compare_ssim(
        fake_np, gt_np,
        channel_axis=-1,
        data_range=1.0
    )
    ssim = torch.tensor(ssim_val, dtype=fake.dtype, device=fake.device)

    return l1_loss, mse_loss, psnr, ssim


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Missing required key, expected one of: {keys}")


def _parse_multitask_inputs(args):
    if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
        batch, outputs = args[0], args[1]
        mask = _first_present(batch, ("mask", "masks", "label"))
        gt = _first_present(batch, ("rebuild", "target", "gt"))
        return (
            mask,
            outputs["x1"],
            outputs["x2"],
            outputs.get("x3", outputs["output"]),
            outputs["output"],
            outputs["mask_logits"],
            gt,
        )

    if len(args) >= 7:
        mask, x_o1, x_o2, x_o3, output, mm, gt = args[:7]
        return mask, x_o1, x_o2, x_o3, output, mm, gt

    raise TypeError("SegMultTaskLoss expects (batch_dict, output_dict) or legacy tensor arguments.")


def _loss_dict(total, **items):
    output = {"total": total}
    output.update(items)
    return output


CE_IGNORE_INDEX = -100


def _sanitize_ce_target(logits, target, *, context):
    target = target.long()
    num_classes = int(logits.shape[1])
    invalid = (target < 0) | (target >= num_classes)
    if not bool(invalid.any().item()):
        return target, True

    invalid_values = [int(v) for v in torch.unique(target[invalid].detach()).cpu().tolist()]
    invalid_pixels = int(invalid.sum().item())
    total_pixels = int(target.numel())
    print(
        f"[LossLabelError] {context}: invalid_values={invalid_values}, "
        f"invalid_pixels={invalid_pixels}/{total_pixels}, num_classes={num_classes}; "
        f"set invalid pixels to ignore_index={CE_IGNORE_INDEX}"
    )

    valid_pixels = total_pixels - invalid_pixels
    target = target.clone()
    target[invalid] = CE_IGNORE_INDEX
    return target, valid_pixels > 0


class SegMultTaskLoss(nn.Module):
    def __init__(self):
        super(SegMultTaskLoss, self).__init__()
        self.l1 = nn.L1Loss()
        weights = torch.tensor([1.0, 4.0, 5.0, 6.0], dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=CE_IGNORE_INDEX)

    def forward(self, *args, **kwargs):
        mask, x_o1, x_o2, x_o3, output, mm, gt = _parse_multitask_inputs(args)
        # 初始化四个掩码，分别对应不同的标签值
        mask0 = (mask == 0).float()  # 标签为0的掩码 背景
        mask1 = (mask == 1).float()  # 标签为1的掩码 手写
        mask2 = (mask == 2).float()  # 标签为2的掩码 打印
        mask3 = (mask == 3).float()  # 标签为3的掩码 重叠
        # print(mask.shape,x_o1.shape,x_o2.shape,x_o3.shape,output.shape,mm.shape,gt.shape)
        # print(mask0.shape,mask1.shape,mask2.shape,mask3.shape)


        # 重建损失
        # holeLoss = 10 * self.l1((1 - mask) * output, (1 - mask) * gt)
        # validAreaLoss = 2 * self.l1(mask * output, mask * gt)
        #print(mask0.shape,mask1.shape,mask2.shape,mask3.shape,output.shape,gt.shape)
        refinement_loss = (2 * self.l1(mask0 * output, mask0 * gt) +
                           8 * self.l1(mask1 * output, mask1 * gt) +
                           10 * self.l1(mask2 * output, mask2 * gt) +
                           12 * self.l1(mask3 * output, mask3 * gt))

        # print(mm.shape,mask.shape)
        # mask_loss = dice_loss(mm,  mask)
        mask_target, has_valid_target = _sanitize_ce_target(
            mm,
            mask.squeeze(1),
            context="SegMultTaskLoss",
        )
        mask_loss = self.ce(mm, mask_target) if has_valid_target else mm.sum() * 0.0

        # 计算多尺度重建损失
        msrloss = (0.8 * self.l1(mask0 * x_o3, mask0 * gt) +
                   4 * self.l1(mask1 * x_o3, mask1 * gt) +
                   5 * self.l1(mask2 * x_o3, mask2 * gt) +
                   6 * self.l1(mask3 * x_o3, mask3 * gt))



        gt = F.interpolate(gt, scale_factor=0.5)
        mask0 = F.interpolate(mask0, scale_factor=0.5)
        mask1 = F.interpolate(mask1, scale_factor=0.5)
        mask2 = F.interpolate(mask2, scale_factor=0.5)
        mask3 = F.interpolate(mask3, scale_factor=0.5)

        msrloss += (1 * self.l1(mask0 * x_o2, mask0 * gt) +
                    3 * self.l1(mask1 * x_o2, mask1 * gt) +
                    4 * self.l1(mask2 * x_o2, mask2 * gt) +
                    5 * self.l1(mask3 * x_o2, mask3 * gt))



        gt = F.interpolate(gt, scale_factor=0.5)
        mask0 = F.interpolate(mask0, scale_factor=0.5)
        mask1 = F.interpolate(mask1, scale_factor=0.5)
        mask2 = F.interpolate(mask2, scale_factor=0.5)
        mask3 = F.interpolate(mask3, scale_factor=0.5)

        msrloss += (0.8 * self.l1(mask0 * x_o1, mask0 * gt) +
                    2 * self.l1(mask1 * x_o1, mask1 * gt) +
                    2 * self.l1(mask2 * x_o1, mask2 * gt) +
                    4 * self.l1(mask3 * x_o1, mask3 * gt))



        # 总损失合成
        # GLoss = msrloss + holeLoss + validAreaLoss + mask_loss
        GLoss = msrloss + refinement_loss + mask_loss

        return _loss_dict(
            GLoss.sum(),
            multiscale=msrloss,
            refinement=refinement_loss,
            mask=mask_loss,
        )


class SegMultTaskLoss3Tags(nn.Module):
    def __init__(self):
        super(SegMultTaskLoss3Tags, self).__init__()
        self.l1 = nn.L1Loss()
        weights = torch.tensor([1.0, 4.0, 5.0], dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=CE_IGNORE_INDEX)

    @staticmethod
    def _to_three_tag_mask(mask):
        return torch.where(mask == 3, torch.ones_like(mask), mask)

    def forward(self, *args, **kwargs):
        mask, x_o1, x_o2, x_o3, output, mm, gt = _parse_multitask_inputs(args)
        mask = self._to_three_tag_mask(mask)
        mask0 = (mask == 0).float()  # 背景
        mask1 = (mask == 1).float()  # 手写；重叠区域也映射到该类
        mask2 = (mask == 2).float()  # 打印

        refinement_loss = (2 * self.l1(mask0 * output, mask0 * gt) +
                           8 * self.l1(mask1 * output, mask1 * gt) +
                           10 * self.l1(mask2 * output, mask2 * gt))

        mask_target, has_valid_target = _sanitize_ce_target(
            mm,
            mask.squeeze(1),
            context="SegMultTaskLoss3Tags",
        )
        mask_loss = self.ce(mm, mask_target) if has_valid_target else mm.sum() * 0.0

        msrloss = (0.8 * self.l1(mask0 * x_o3, mask0 * gt) +
                   4 * self.l1(mask1 * x_o3, mask1 * gt) +
                   5 * self.l1(mask2 * x_o3, mask2 * gt))

        gt = F.interpolate(gt, scale_factor=0.5)
        mask0 = F.interpolate(mask0, scale_factor=0.5)
        mask1 = F.interpolate(mask1, scale_factor=0.5)
        mask2 = F.interpolate(mask2, scale_factor=0.5)

        msrloss += (1 * self.l1(mask0 * x_o2, mask0 * gt) +
                    3 * self.l1(mask1 * x_o2, mask1 * gt) +
                    4 * self.l1(mask2 * x_o2, mask2 * gt))

        gt = F.interpolate(gt, scale_factor=0.5)
        mask0 = F.interpolate(mask0, scale_factor=0.5)
        mask1 = F.interpolate(mask1, scale_factor=0.5)
        mask2 = F.interpolate(mask2, scale_factor=0.5)

        msrloss += (0.8 * self.l1(mask0 * x_o1, mask0 * gt) +
                    2 * self.l1(mask1 * x_o1, mask1 * gt) +
                    2 * self.l1(mask2 * x_o1, mask2 * gt))

        GLoss = msrloss + refinement_loss + mask_loss

        return _loss_dict(
            GLoss.sum(),
            multiscale=msrloss,
            refinement=refinement_loss,
            mask=mask_loss,
        )





class RefinementLoss(nn.Module):
    def __init__(self):
        super(RefinementLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            batch, outputs = args[0], args[1]
            mask = _first_present(batch, ("mask", "masks", "label"))
            gt = _first_present(batch, ("rebuild", "target", "gt"))
            output = outputs["output"]
        elif len(args) >= 3:
            mask, output, gt = args[:3]
        else:
            raise TypeError("RefinementLoss expects (batch_dict, output_dict) or legacy tensor arguments.")
        b, _, width, height = mask.shape

        # 创建 4 个单通道掩码
        mask0 = (mask == 0).float()  # 背景
        mask1 = (mask == 1).float()  # 打印
        mask2 = (mask == 2).float()  # 手写
        mask3 = (mask == 3).float()  # 重叠

        mask0 = mask0.expand(b, 3, width, height)
        mask1 = mask1.expand(b, 3, width, height)
        mask2 = mask2.expand(b, 3, width, height)
        mask3 = mask3.expand(b, 3, width, height)

        refinement_loss = (2 * self.l1(mask0 * output, mask0 * gt) + 8 * self.l1(mask1 * output, mask1 * gt) +
                           10 * self.l1( mask2 * output, mask2 * gt) + 12 * self.l1(mask3 * output, mask3 * gt))

        return _loss_dict(refinement_loss, refinement=refinement_loss)


class RefinementLoss3Tags(nn.Module):
    def __init__(self):
        super(RefinementLoss3Tags, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            batch, outputs = args[0], args[1]
            mask = _first_present(batch, ("mask", "masks", "label"))
            gt = _first_present(batch, ("rebuild", "target", "gt"))
            output = outputs["output"]
        elif len(args) >= 3:
            mask, output, gt = args[:3]
        else:
            raise TypeError("RefinementLoss3Tags expects (batch_dict, output_dict) or legacy tensor arguments.")
        mask = torch.where(mask == 3, torch.ones_like(mask), mask)
        b, _, width, height = mask.shape

        mask0 = (mask == 0).float().expand(b, 3, width, height)
        mask1 = (mask == 1).float().expand(b, 3, width, height)
        mask2 = (mask == 2).float().expand(b, 3, width, height)

        refinement_loss = (2 * self.l1(mask0 * output, mask0 * gt) +
                           8 * self.l1(mask1 * output, mask1 * gt) +
                           10 * self.l1(mask2 * output, mask2 * gt))

        return _loss_dict(refinement_loss, refinement=refinement_loss)


class DistillationLoss(nn.Module):
    """
    Distill only final image output and mask logits with L2(MSE) loss.
    """
    def __init__(self):
        super(DistillationLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            student_outputs, teacher_outputs = args[0], args[1]
            student_output = student_outputs["output"]
            student_mm = student_outputs["mask_logits"]
            teacher_output = teacher_outputs["output"]
            teacher_mm = teacher_outputs["mask_logits"]
        else:
            student_output = kwargs.get("student_output", args[0] if len(args) > 0 else None)
            student_mm = kwargs.get("student_mm", args[1] if len(args) > 1 else None)
            teacher_output = kwargs.get("teacher_output", args[2] if len(args) > 2 else None)
            teacher_mm = kwargs.get("teacher_mm", args[3] if len(args) > 3 else None)

        teacher_output = teacher_output.detach()
        teacher_mm = teacher_mm.detach()

        output_loss = self.mse(student_output, teacher_output)
        mask_loss = self.mse(student_mm, teacher_mm)
        return _loss_dict(output_loss + mask_loss, output=output_loss, mask=mask_loss)


def deep_feature_l2_loss(student_deep, teacher_deep):
    return F.mse_loss(student_deep, teacher_deep)


def deep_feature_at_loss(student_deep, teacher_deep, eps=1e-12):
    """
    Attention Transfer:
      A(F) = normalize(mean(F^2, dim=1), p=2)
    """
    teacher_deep = teacher_deep.detach()
    att_s = torch.mean(student_deep.pow(2), dim=1, keepdim=True)
    att_t = torch.mean(teacher_deep.pow(2), dim=1, keepdim=True)

    att_s = att_s.flatten(1)
    att_t = att_t.flatten(1)

    att_s = att_s / (att_s.norm(p=2, dim=1, keepdim=True) + eps)
    att_t = att_t / (att_t.norm(p=2, dim=1, keepdim=True) + eps)

    return F.mse_loss(att_s, att_t)
