import glob
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 定义检测函数
def test_maskToTensor(mask_tensor):
    color_to_label = {
        (255, 255, 255): 0,  # white -> 0
        (0, 255, 0): 1,  # green -> 1
        (255, 0, 0): 2,  # red -> 2
        (0, 0, 255): 3  # blue -> 3
    }

    label_to_color = {v: k for k, v in color_to_label.items()}  # 反向映射

    """
    这个函数将检测 maskToTensor 是否正确。
    输入一个单通道的三维张量，生成一个三通道的图像并展示。
    """


    # Step 2: 将 tensor 转换回颜色图像
    mask_array = mask_tensor.squeeze(0).numpy()  # 去掉通道维度，变为 (H, W)

    # Step 3: 根据标签生成对应的颜色图像
    height, width = mask_array.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 为每个标签像素填充对应的颜色
    for label, color in label_to_color.items():
        color_image[mask_array == label] = color

    # Step 4: 使用 PIL.Image 展示图像
    img = Image.fromarray(color_image)
    img.show()

def maskToTensor(mask):
    mask_array = np.array(mask)

    # Create an empty single-channel tensor (initialized to zeros)
    mask_tensor = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.int64)  # 单通道 tensor

    # Color-to-label mapping
    color_to_label = {
        (255, 255, 255): 0,  # white -> 0
        (0, 255, 0): 1,  # green -> 1
        (255, 0, 0): 2,  # red -> 2
        (0, 0, 255): 3  # blue -> 3
    }

    # Iterate through the color-to-label mapping and set values
    for color, label in color_to_label.items():
        mask_tensor[(mask_array[:, :, 0] == color[0]) &
                    (mask_array[:, :, 1] == color[1]) &
                    (mask_array[:, :, 2] == color[2])] = label  # Assign label to the corresponding pixel

    # Convert the numpy array to a tensor
    return torch.tensor(mask_tensor, dtype=torch.long)  # Convert to long type tensor (suitable for classification)

class SegImageDataset(Dataset):
    def __init__(self, root, mode="train"):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 获取 'images'、'rebuild' 和 'mask' 文件夹路径
        image_dir = os.path.join(root, 'images')
        rebuild_dir = os.path.join(root, 'rebuild')
        mask_dir = os.path.join(root, 'mask')
        print(image_dir, rebuild_dir, mask_dir)
        # 获取 images 文件夹中所有 png 文件的文件名（去除扩展名）
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        self.image_names = [os.path.splitext(os.path.basename(f))[0] for f in image_files]

        # 创建一个字典，存储 rebuild 和 mask 文件的路径，以文件名（去除扩展名）为键
        self.image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in
                             glob.glob(os.path.join(image_dir, "*.png"))}
        self.rebuild_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in
                             glob.glob(os.path.join(rebuild_dir, "*.png"))}
        self.mask_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in
                          glob.glob(os.path.join(mask_dir, "*.png"))}

        # 过滤出在三个文件夹中都有对应文件的文件名
        self.valid_image_names = [name for name in self.image_names if
                                  name in self.rebuild_dict and name in self.mask_dict]

        print(f"{mode} image files: {len(self.valid_image_names)}")  # 打印有效文件数量

    def __getitem__(self, index):
        # 获取当前索引对应的文件名
        file_name = self.valid_image_names[index]

        # 加载 images, rebuild 和 mask 文件
        img_A = Image.open(self.image_dict[file_name])
        img_B = Image.open(self.rebuild_dict[file_name])
        img_C = Image.open(self.mask_dict[file_name])

        if np.random.random() < 0.5:
            img_A, img_B, img_C = random_filp(img_A, img_B, img_C)

        # 应用预定义的转换（比如标准化、归一化等）
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        mask = maskToTensor(img_C).unsqueeze(0)
        #test_maskToTensor(mask)
        return img_A,img_B,mask
        #return {"A": img_A, "B": img_B, "mask": mask}


    def __len__(self):
        # 返回有效文件的数量
        return len(self.valid_image_names)

def random_filp(image, rebuild, mask):
    type = np.random.random()
    if type < 0.2:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        rebuild = rebuild.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif type < 0.4:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        rebuild = rebuild.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif type < 0.6:
        image = image.transpose(Image.Transpose.ROTATE_90)
        rebuild = rebuild.transpose(Image.Transpose.ROTATE_90)
        mask = mask.transpose(Image.Transpose.ROTATE_90)
    elif type < 0.8:
        image = image.transpose(Image.Transpose.ROTATE_180)
        rebuild = rebuild.transpose(Image.Transpose.ROTATE_180)
        mask = mask.transpose(Image.Transpose.ROTATE_180)
    elif type < 1:
        image = image.transpose(Image.Transpose.ROTATE_270)
        rebuild = rebuild.transpose(Image.Transpose.ROTATE_270)
        mask = mask.transpose(Image.Transpose.ROTATE_270)
    return image, rebuild, mask

if __name__ == "__main__":
    dataset = DataLoader(
        SegImageDataset("../dataset/demo"),
        batch_size=1,
    )
    for idx, (imgs, gt, masks) in enumerate(dataset):
        print(imgs.shape, gt.shape, masks.shape)