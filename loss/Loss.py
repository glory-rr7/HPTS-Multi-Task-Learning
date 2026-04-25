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
    mean=(0.485, 0.456, 0.406),     # ImageNet 默认
    std=(0.229, 0.224, 0.225),
    SSIM=True,
):
    """
    fake / gt : 4-D tensor  [B, C, H, W]，已做过 ImageNet Normalize。
    先“反标准化”回 0-1 再计算 L1 / MSE / PSNR / SSIM。
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

class SegMultTaskLoss(nn.Module):
    def __init__(self):
        super(SegMultTaskLoss, self).__init__()
        self.l1 = nn.L1Loss()
        weights = torch.tensor([1.0, 4.0, 5.0, 6.0], dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, mask, x_o1, x_o2, x_o3, output, mm, gt, idx):
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
        mask_loss = self.ce(mm, mask.squeeze(1))

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

        return GLoss.sum()


class SegMultTaskLoss3Tags(nn.Module):
    def __init__(self):
        super(SegMultTaskLoss3Tags, self).__init__()
        self.l1 = nn.L1Loss()
        weights = torch.tensor([1.0, 4.0, 5.0], dtype=torch.float32)
        self.ce = nn.CrossEntropyLoss(weight=weights)

    @staticmethod
    def _to_three_tag_mask(mask):
        return torch.where(mask == 3, torch.ones_like(mask), mask)

    def forward(self, mask, x_o1, x_o2, x_o3, output, mm, gt, idx):
        mask = self._to_three_tag_mask(mask)
        mask0 = (mask == 0).float()  # 背景
        mask1 = (mask == 1).float()  # 手写；重叠区域也映射到该类
        mask2 = (mask == 2).float()  # 打印

        refinement_loss = (2 * self.l1(mask0 * output, mask0 * gt) +
                           8 * self.l1(mask1 * output, mask1 * gt) +
                           10 * self.l1(mask2 * output, mask2 * gt))

        mask_loss = self.ce(mm, mask.squeeze(1))

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

        return GLoss.sum()





class RefinementLoss(nn.Module):
    def __init__(self):
        super(RefinementLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, gt, ):
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

        return refinement_loss


class RefinementLoss3Tags(nn.Module):
    def __init__(self):
        super(RefinementLoss3Tags, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, mask, output, gt, ):
        mask = torch.where(mask == 3, torch.ones_like(mask), mask)
        b, _, width, height = mask.shape

        mask0 = (mask == 0).float().expand(b, 3, width, height)
        mask1 = (mask == 1).float().expand(b, 3, width, height)
        mask2 = (mask == 2).float().expand(b, 3, width, height)

        refinement_loss = (2 * self.l1(mask0 * output, mask0 * gt) +
                           8 * self.l1(mask1 * output, mask1 * gt) +
                           10 * self.l1(mask2 * output, mask2 * gt))

        return refinement_loss


class DistillationLoss(nn.Module):
    """
    Distill only final image output and mask logits with L2(MSE) loss.
    """
    def __init__(self):
        super(DistillationLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_output, student_mm, teacher_output, teacher_mm):
        teacher_output = teacher_output.detach()
        teacher_mm = teacher_mm.detach()

        output_loss = self.mse(student_output, teacher_output)
        mask_loss = self.mse(student_mm, teacher_mm)
        return output_loss, mask_loss


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
