import glob
import os
import random
from os import PathLike

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image

from dataloader.augmentation import apply_random_augmentation, apply_strategy_shared_aug, apply_strategy_extra_aug
from dataloader.label_utils import decode_segmentation_label, resolve_dataset_directory


def _normalize_roots(root):
    """Accept str or yaml list for dataset roots."""
    if isinstance(root, (str, PathLike)):
        roots = [os.fspath(root)]
    elif isinstance(root, (list, tuple)):
        roots = [os.fspath(p) for p in root if isinstance(p, (str, PathLike)) and str(p).strip()]
    else:
        raise TypeError("`root` must be a string path or a list/tuple of paths.")

    if not roots:
        raise ValueError("No valid dataset path is provided in `root`.")

    return roots


def _print_if_label_out_of_range(label, label_path, *, context):
    if label.numel() == 0:
        print(f"[DatasetLabelError] {context}: empty label, path={label_path}")
        return
    label_min = int(label.min().item())
    label_max = int(label.max().item())
    if label_min < 0 or label_max > 3:
        print(f"[DatasetLabelError] {context}: path={label_path}")


class BlockSegImageDataset(Dataset):
    def __init__(self, root, mode="train", tile_size=512, use_aug=False, aug_prob=0.6, use_strategy=False):
        self.normalize = T.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
        )
        self.tile_size = tile_size
        self.use_aug = use_aug
        self.aug_prob = aug_prob
        self.use_strategy = use_strategy
        self.roots = _normalize_roots(root)

        self.samples = []
        for root_path in self.roots:
            image_dir = resolve_dataset_directory(root_path, 'images', 'crop')
            rebuild_dir = os.path.join(root_path, 'rebuild')
            label_dir = resolve_dataset_directory(root_path, 'labels', 'label')

            image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
            rebuild_dict = {
                os.path.splitext(os.path.basename(f))[0]: f
                for f in glob.glob(os.path.join(rebuild_dir, "*.png"))
            }
            label_dict = {
                os.path.splitext(os.path.basename(f))[0]: f
                for f in glob.glob(os.path.join(label_dir, "*.png"))
            }

            valid_count = 0
            for image_path in image_files:
                name = os.path.splitext(os.path.basename(image_path))[0]
                rebuild_path = rebuild_dict.get(name)
                label_path = label_dict.get(name)
                if rebuild_path is None or label_path is None:
                    continue
                self.samples.append({
                    "image_path": image_path,
                    "rebuild_path": rebuild_path,
                    "label_path": label_path,
                })
                valid_count += 1

            print(f"{mode} image files in {root_path}: {valid_count}")

        if not self.samples:
            raise RuntimeError(f"No valid samples found for mode={mode}, roots={self.roots}.")

        self.tiles_info = []
        self._prepare_tiles()

        print(f"{mode} total image files: {len(self.samples)}")
        print(f"Total tiles: {len(self.tiles_info)}")

    def _prepare_tiles(self):
        """为每个图像分割成小块并记录每个小块的位置信息"""
        for sample in self.samples:
            image_path = sample['image_path']
            rebuild_path = sample['rebuild_path']
            label_path = sample['label_path']

            # 只读尺寸，不解码像素
            with Image.open(image_path) as image:
                img_width, img_height = image.size

            positions = self._split_image(img_width, img_height)

            for pos in positions:
                self.tiles_info.append({
                    'image_path': image_path,
                    'rebuild_path': rebuild_path,
                    'label_path': label_path,
                    'position': pos
                })

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
                if left < 0 or top < 0 or right >= width or bottom >= height:
                    continue
                if right - left + 1 <= 0 or bottom - top + 1 <= 0:
                    continue
                positions.append([left, top, right, bottom])
        return positions

    def __len__(self):
        return len(self.tiles_info)

    def __getitem__(self, idx):
        info = self.tiles_info[idx]
        left, top, right, bottom = info['position']
        h, w = bottom - top + 1, right - left + 1

        img = read_image(info['image_path'])      # (C, H, W) uint8
        reb = read_image(info['rebuild_path'])
        label = read_image(info['label_path'])     # (1, H, W) 灰度，值 0~3

        # 处理可能的 RGBA
        if img.shape[0] == 4:
            img = img[:3]
        if reb.shape[0] == 4:
            reb = reb[:3]
        label = decode_segmentation_label(label, info['label_path'])

        # tensor 切片裁剪 tile
        img = img[:, top:top + h, left:left + w]
        reb = reb[:, top:top + h, left:left + w]
        label = label[:, top:top + h, left:left + w]

        if self.use_strategy:
            # Strategy 模式：生成共享几何变换的两个视角
            data = {"image": img, "mask": label.to(dtype=torch.uint8), "rebuild": reb}

            # 共享几何 aug（flip/rot + wave stretch），ori 和 aug 视角共用
            data = apply_strategy_shared_aug(data)

            # ori 视角：直接 normalize
            ori_img = self.normalize(data["image"].float() / 255.0)
            ori_reb = self.normalize(data["rebuild"].float() / 255.0)
            ori_label = data["mask"].long()

            # aug 视角：在共享 aug 基础上叠加 extra aug
            aug_data = apply_strategy_extra_aug(dict(data))
            aug_img = self.normalize(aug_data["image"].float() / 255.0)
            aug_reb = self.normalize(aug_data["rebuild"].float() / 255.0)
            aug_label = aug_data["mask"].long()
            tile_context = f"BlockSegImageDataset/strategy/tile={info['position']}"
            _print_if_label_out_of_range(ori_label, info["label_path"], context=f"{tile_context}/ori")
            _print_if_label_out_of_range(aug_label, info["label_path"], context=f"{tile_context}/aug")

            return ori_img, ori_reb, ori_label, aug_img, aug_reb, aug_label

        # 普通模式
        # 随机翻转/旋转
        if random.random() < 0.5:
            img, reb, label = random_flip(img, reb, label)

        if self.use_aug:
            sample = {"image": img, "mask": label, "rebuild": reb}
            sample = apply_random_augmentation(sample, enabled=True, apply_prob=self.aug_prob)
            img, label, reb = sample["image"], sample["mask"], sample["rebuild"]

        # uint8 -> float32 [0,1] -> normalize
        img = self.normalize(img.float() / 255.0)
        reb = self.normalize(reb.float() / 255.0)
        label = label.long()  # (1, H, W)，值 ∈ {0,1,2,3}
        _print_if_label_out_of_range(
            label,
            info["label_path"],
            context=f"BlockSegImageDataset/tile={info['position']}",
        )

        return img, reb, label


def random_flip(img, reb, label):
    """对 (C, H, W) tensor 做同步随机翻转/旋转"""
    r = random.random()
    if r < 0.2:
        img = torch.flip(img, dims=[2])
        reb = torch.flip(reb, dims=[2])
        label = torch.flip(label, dims=[2])
    elif r < 0.4:
        img = torch.flip(img, dims=[1])
        reb = torch.flip(reb, dims=[1])
        label = torch.flip(label, dims=[1])
    elif r < 0.6:
        img = torch.rot90(img, k=1, dims=[1, 2])
        reb = torch.rot90(reb, k=1, dims=[1, 2])
        label = torch.rot90(label, k=1, dims=[1, 2])
    elif r < 0.8:
        img = torch.rot90(img, k=2, dims=[1, 2])
        reb = torch.rot90(reb, k=2, dims=[1, 2])
        label = torch.rot90(label, k=2, dims=[1, 2])
    else:
        img = torch.rot90(img, k=3, dims=[1, 2])
        reb = torch.rot90(reb, k=3, dims=[1, 2])
        label = torch.rot90(label, k=3, dims=[1, 2])
    return img, reb, label


if __name__ == "__main__":
    dataset = DataLoader(
        BlockSegImageDataset("../dataset/truedata_test/processed3", tile_size=256),
        batch_size=1,
        shuffle=True,
    )
    for idx, (imgs, gt, labels) in enumerate(dataset):
        print(imgs.shape, gt.shape, labels.shape)
