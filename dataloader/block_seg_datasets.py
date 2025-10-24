import glob
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.io import read_image
import torchvision.transforms as T
import torch



class BlockSegImageDataset(Dataset):
    def __init__(self, root,  mode="train", tile_size=512):
        self.transform = T.Compose([
            # 改成直接转 float32，不再二次 /255
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406,],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.keys = torch.tensor([0xFFFFFF, 0x00FF00, 0xFF0000, 0x0000FF], dtype=torch.int32)
        self.vals = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        self.tile_size = tile_size

        # 获取 'images'、'rebuild' 和 'mask' 文件夹路径
        image_dir = os.path.join(root, 'images')
        rebuild_dir = os.path.join(root, 'rebuild')
        mask_dir = os.path.join(root, 'mask')

        # 获取 images 文件夹中所有 png 文件的文件名（去除扩展名）
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.image_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

        # 创建一个字典，存储 image、rebuild 和 mask 文件的路径，以文件名（去除扩展名）为键
        self.image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in
                             glob.glob(os.path.join(image_dir, "*.png"))}
        self.rebuild_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in
                             glob.glob(os.path.join(rebuild_dir, "*.png"))}
        self.mask_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in
                          glob.glob(os.path.join(mask_dir, "*.png"))}

        # 过滤出在三个文件夹中都有对应文件的文件名
        self.valid_image_names = [name for name in self.image_names if
                                  name in self.rebuild_dict and name in self.mask_dict]

        # 存储每个图像的分割块信息 (包括图像路径、重建路径、掩码路径、分割位置)
        self.tiles_info = []
        self._prepare_tiles()

        print(f"{mode} image files: {len(self.valid_image_names)}")  # 打印有效文件数量
        print(f"Total tiles: {len(self.tiles_info)}")  # 打印总的分割块数量

    def mask_to_label(self, m: torch.Tensor) -> torch.Tensor:
        m = m.to(torch.int32)          # ★ 修正 1
        packed = (m[0] << 16) | (m[1] << 8) | m[2]
        lbl = torch.zeros_like(packed, dtype=torch.long)
        for k, v in zip(self.keys, self.vals):
            lbl.masked_fill_(packed == k, v)
        return lbl
    def _prepare_tiles(self):
        """为每个图像分割成小块并记录每个小块的位置信息"""
        for image_name in self.valid_image_names:
            image_path = self.image_dict[image_name]
            rebuild_path = self.rebuild_dict[image_name]
            mask_path = self.mask_dict[image_name]

            # 打开图像并获取尺寸
            image = Image.open(image_path)
            img_width, img_height = image.size

            # 使用 split_image 函数分割图像并记录每个块的位置信息
            positions = self._split_image(img_width, img_height)

            for pos in positions:
                self.tiles_info.append({
                    'image_path': image_path,
                    'rebuild_path': rebuild_path,
                    'mask_path': mask_path,
                    'position': pos
                })

        # valid_tiles = []
        # for info in self.tiles_info:
        #    msk = read_image(info['mask_path'])
        #    left, top, right, bottom = info['position']
        #    msk_crop = msk[:, top:bottom+1, left:right+1]
        #    mask_lbl = self.mask_to_label(msk_crop)
        #    if (mask_lbl >= 1).any():
        #        valid_tiles.append(info)
        # self.tiles_info = valid_tiles

    def _split_image(self, width, height):
        """将图像分割成指定大小的小块，只保留边界合法的patch"""
        top_list = []
        left_list = []
        for i in range(0, height, self.tile_size):
            top_list.append(i)
        if height % self.tile_size != 0:
            top_list.pop()
            top_list.append(height - self.tile_size)
        for j in range(0, width, self.tile_size):
            left_list.append(j)
        if width % self.tile_size != 0:
            left_list.pop()
            left_list.append(width - self.tile_size)

        positions = []
        for top in top_list:
            for left in left_list:
                bottom = min(top + self.tile_size - 1, height - 1)
                right = min(left + self.tile_size - 1, width - 1)
                # 检查所有坐标合法
                if left < 0 or top < 0 or right >= width or bottom >= height:
                    continue  # 跳过非法patch
                # 宽高为负或0也跳过
                if right - left + 1 <= 0 or bottom - top + 1 <= 0:
                    continue
                positions.append([left, top, right, bottom])
        return positions

    # def _split_image(self, width, height):
    #     """将图像分割成指定大小的小块"""
    #     top_list = []
    #     left_list = []
    #     for i in range(0, height, self.tile_size):
    #         top_list.append(i)
    #     if height % self.tile_size != 0:
    #         top_list.pop()
    #         top_list.append(height - self.tile_size)
    #     for j in range(0, width, self.tile_size):
    #         left_list.append(j)
    #     if width % self.tile_size != 0:
    #         left_list.pop()
    #         left_list.append(width - self.tile_size)
    #
    #     positions = []
    #     for top in top_list:
    #         for left in left_list:
    #             bottom = min(top + self.tile_size - 1, height)
    #             right = min(left + self.tile_size - 1, width)
    #             positions.append([left, top, right, bottom])
    #
    #     return positions

    def __len__(self):
        return len(self.tiles_info)

    def __getitem__(self, idx):
        info = self.tiles_info[idx]
        left, top, right, bottom = info['position']
        h, w = bottom - top + 1, right - left + 1

        # 1) 直接读成 Tensor (C×H×W)，速度比 PIL 快
        img = read_image(info['image_path'])  # uint8 tensor
        reb = read_image(info['rebuild_path'])
        msk = read_image(info['mask_path'])
        if img.shape[0] == 4:
            img = img[:3, :, :]
        if reb.shape[0] == 4:
            reb = reb[:3, :, :]
        if msk.shape[0] == 4:
            msk = msk[:3, :, :]
                # 2) tensor 切片代替 crop
        img = img[:, top:top + h, left:left + w]
        reb = reb[:, top:top + h, left:left + w]
        msk = msk[:, top:top + h, left:left + w]

        # 3) 随机水平翻转
        if torch.rand(1) < 0.5:
            img, reb, msk = random_flip(img, reb, msk)

        # 4) 做你已有的 transform（如果你的 transform 接受 PIL，需要改成接受 tensor）
        img = self.transform(img)
        reb = self.transform(reb)
        msk = self.mask_to_label(msk).unsqueeze(0)

        return img, reb, msk

import torch

def random_flip(img, reb, msk):
    """
    img, reb, msk: Tensor of shape (C, H, W), dtype float or uint8
    返回同样 shape 的三张图，做随机翻转/旋转。
    """
    r = torch.rand(1).item()
    if r < 0.2:
        # 水平翻转
        img = torch.flip(img, dims=[2])
        reb = torch.flip(reb, dims=[2])
        msk = torch.flip(msk, dims=[2])
    elif r < 0.4:
        # 垂直翻转
        img = torch.flip(img, dims=[1])
        reb = torch.flip(reb, dims=[1])
        msk = torch.flip(msk, dims=[1])
    elif r < 0.6:
        # 90° 逆时针
        img = torch.rot90(img, k=1, dims=[1, 2])
        reb = torch.rot90(reb, k=1, dims=[1, 2])
        msk = torch.rot90(msk, k=1, dims=[1, 2])
    elif r < 0.8:
        # 180°
        img = torch.rot90(img, k=2, dims=[1, 2])
        reb = torch.rot90(reb, k=2, dims=[1, 2])
        msk = torch.rot90(msk, k=2, dims=[1, 2])
    else:
        # 270° (或 -90°)
        img = torch.rot90(img, k=3, dims=[1, 2])
        reb = torch.rot90(reb, k=3, dims=[1, 2])
        msk = torch.rot90(msk, k=3, dims=[1, 2])
    return img, reb, msk



if __name__ == "__main__":
    dataset = DataLoader(
        BlockSegImageDataset("../dataset/truedata_test/processed3",tile_size=256),
        batch_size=1,
        shuffle=True,

    )
    output_dir = "./output"

    for idx, (imgs, gt, masks) in enumerate(dataset):
        print(imgs.shape, gt.shape, masks.shape)
        pass
        #print(imgs,  gt, masks)

        #break


# import os
# from pathlib import Path
#
# import torch
# from torchvision.transforms.functional import to_pil_image
#
# # ---------- 颜色映射 ----------
# LABEL2COLOR = torch.tensor([
#     [255, 255, 255],   # 0 → white
#     [0,   255,   0],   # 1 → green
#     [255,   0,   0],   # 2 → red
#     [0,     0, 255],   # 3 → blue
# ], dtype=torch.uint8)          # shape (4, 3)
#
# # ---------- 反归一化（按你的 mean/std） ----------
# _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
# _STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
#
#
# @torch.no_grad()
# def save_samples(dataloader, save_dir="vis", batches=None):
#     """
#     将 dataloader 中的 (imgs, gt, masks) 保存为 PNG：
#       • imgs, gt 反归一化后保存为 RGB
#       • masks 0/1/2/3 → 白/绿/红/蓝 伪彩色保存
#     Args
#     ----
#     dataloader : torch.utils.data.DataLoader
#     save_dir   : 根目录，自动新建  img_x.png / gt_x.png / mask_x.png
#     batches    : 仅保存前 N 个 batch，None=全部
#     """
#     Path(save_dir).mkdir(exist_ok=True, parents=True)
#
#     n_saved = 0
#     for batch_idx, (imgs, gt, masks) in enumerate(dataloader):
#         if batches is not None and batch_idx >= batches:
#             break
#
#         # 反归一化到 [0,1]
#         imgs_denorm = (imgs * _STD + _MEAN).clamp(0, 1)
#         gt_denorm   = (gt   * _STD + _MEAN).clamp(0, 1)
#
#         B = imgs.size(0)
#         for i in range(B):
#             # --------- RGB ----------
#             to_pil_image(imgs_denorm[i]).save(f"{save_dir}/img_{n_saved}.png")
#             to_pil_image(gt_denorm[i]).save(f"{save_dir}/gt_{n_saved}.png")
#
#             # --------- Mask ----------
#             m = masks[i, 0]                          # (H, W)
#             color = LABEL2COLOR[m]                   # (H, W, 3), uint8
#             to_pil_image(color.permute(2, 0, 1)).save(f"{save_dir}/mask_{n_saved}.png")
#
#             n_saved += 1
#
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#
#     dl = DataLoader(BlockSegImageDataset("../dataset/demo_none_fixed_size"),
#                     batch_size=4, shuffle=False)
#
#     save_samples(dl, save_dir="debug_vis", batches=1)
#     print("Done! Images are in ./debug_vis/")