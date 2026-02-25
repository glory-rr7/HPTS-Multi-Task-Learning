import glob
import os
import random
from os import PathLike

import torch
import torchvision.transforms as transforms
import torchvision.io as tvio
from torch.utils.data import Dataset, DataLoader

from dataloader.augmentation import apply_random_augmentation


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


class SegImageDataset(Dataset):
    def __init__(self, root, mode="train", use_aug=False, aug_prob=0.6):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.use_aug = use_aug
        self.aug_prob = aug_prob
        self.roots = _normalize_roots(root)

        self.samples = []
        for root_path in self.roots:
            image_dir = os.path.join(root_path, 'images')
            rebuild_dir = os.path.join(root_path, 'rebuild')
            label_dir = os.path.join(root_path, 'labels')
            print(image_dir, rebuild_dir, label_dir)

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
                    "image": image_path,
                    "rebuild": rebuild_path,
                    "label": label_path,
                })
                valid_count += 1

            print(f"{mode} image files in {root_path}: {valid_count}")

        if not self.samples:
            raise RuntimeError(f"No valid samples found for mode={mode}, roots={self.roots}.")

        print(f"{mode} total image files: {len(self.samples)}")

    def __getitem__(self, index):
        sample = self.samples[index]

        # read_image 直接返回 (C, H, W) uint8 tensor，比 PIL 快
        img = tvio.read_image(sample["image"])      # (3, H, W)
        reb = tvio.read_image(sample["rebuild"])     # (3, H, W)
        label = tvio.read_image(sample["label"])     # (1, H, W) 灰度，值 0~3

        # 处理可能的 4 通道（RGBA）
        if img.shape[0] == 4:
            img = img[:3]
        if reb.shape[0] == 4:
            reb = reb[:3]
        # label 只取第一通道
        if label.shape[0] > 1:
            label = label[0:1]

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

        return img, reb, label

    def __len__(self):
        return len(self.samples)


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
        SegImageDataset("../dataset/demo"),
        batch_size=1,
    )
    for idx, (imgs, gt, labels) in enumerate(dataset):
        print(imgs.shape, gt.shape, labels.shape)
