import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import random

def maskToTensor(mask,device):
    # Define the color-to-label and label-to-color mappings
    color_to_label = {
        (255, 255, 255): 0,  # white -> 0
        (0, 255, 0): 1,  # green -> 1
        (255, 0, 0): 2,  # red -> 2
        (0, 0, 255): 3  # blue -> 3
    }
    label_to_color = {v: k for k, v in color_to_label.items()}  # Reverse mapping

    # Check if mask is a single channel and convert it to color if necessary
    if mask.dim() == 4 and mask.shape[1] == 1:  # Mask is a single channel with batch size
        batch_size, _, H, W = mask.size()
        mask_color = torch.zeros(batch_size, 3, H, W, device=device)
        for label_val, color in label_to_color.items():
            # Convert color to [0,1] range
            color_tensor = torch.tensor(color, device=device, dtype=torch.float32) / 255.0
            # Create boolean mask for current label
            label_mask = (mask.squeeze(1) == label_val).float()
            # Apply color to each channel
            mask_color[:, 0] += label_mask * color_tensor[0]  # Red channel
            mask_color[:, 1] += label_mask * color_tensor[1]  # Green channel
            mask_color[:, 2] += label_mask * color_tensor[2]  # Blue channel
    else:
        # If mask is already in color, just use it as is
        mask_color = mask
        mask_color = (mask_color - mask_color.min()) / (mask_color.max() - mask_color.min())

    return mask_color

def sample_images(valdataset, model, save_dir):
    """Saves a generated sample from the validation set as a grid of 5 rows and 5 columns."""
    # Initialize a list to store each row of images
    rows = []
    # Get the device where the model is located
    device = next(model.parameters()).device
    model.eval()


    # Iterate 5 times to generate 5 rows
    for _ in range(5):
        idx = random.randint(0, len(valdataset) - 1)
        real_A, real_B, mask = valdataset[idx]
        real_A = real_A.unsqueeze(0).to(device)  # 加 batch 维度
        real_B = real_B.unsqueeze(0).to(device)  # 加 batch 维度
        mask = mask.unsqueeze(0).to(device)  # 加 batch 维度
        # Move tensors to the same device as the model
        real_A = real_A.to(device)
        real_B = real_B.to(device)
        mask = mask.to(device)

        if model.return_num() == 1:
            output = model(real_A)
            label = mask
        elif model.return_num() == 4:
            _, _, output, label = model(real_A)
        else:
            _, _, _, output, label = model(real_A)


        label = torch.argmax(label, 1)
        label = label.unsqueeze(0)


        real_A = (real_A - real_A.min()) / (real_A.max() - real_A.min())
        real_B = (real_B - real_B.min()) / (real_B.max() - real_B.min())
        output = (output - output.min()) / (output.max() - output.min())
        mask = maskToTensor(mask,device)
        label = maskToTensor(label,device)
        # Concatenate images
        #print(real_A.shape,real_B.shape,output.shape,mask.shape,label.shape)
        row = torch.cat((real_A.data, output.data, real_B.data, label.data, mask.data), -1)
        rows.append(row)

    # Concatenate all rows vertically to form a large image
    large_image = torch.cat(rows, -2)

    model.train()

    save_image(large_image, save_dir, nrow=1, normalize=True)
    print(f"Image saved to {save_dir}")

def save_tensor_as_image(tensor, filename):
    """将PyTorch张量保存为图像文件

    参数：
        tensor (torch.Tensor): 输入张量，支持形状：
            - (H, W)         灰度图
            - (C, H, W)      彩色图（自动检测3/4通道）
            - (1, C, H, W)   带批次维度的彩色图
            - (B, C, H, W)   仅当B=1时支持
        filename (str): 输出文件名（需包含扩展名）

    支持数据类型：
        - torch.uint8       直接保存
        - torch.float       自动归一化（假设输入范围0-1）
    """
    # 移除梯度追踪并转CPU
    tensor = tensor.detach().cpu()

    # 处理批次维度
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError("only save single image file")
        tensor = tensor.squeeze(0)

    # 处理通道维度 (C, H, W) -> (H, W, C)
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)

    # 转换为numpy数组
    arr = tensor.numpy()

    # 处理数据类型
    if arr.dtype in [np.float32, np.float64]:
        arr = np.clip(arr, 0, 1)  # 确保数值范围正确
        arr = (arr * 255).astype(np.uint8)
    elif arr.dtype == np.uint8:
        pass  # 保持原样
    else:
        raise TypeError(f"error type: {arr.dtype}")

    # 处理单通道维度
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)

    # 自动检测图像模式
    if arr.ndim == 2:
        mode = 'L'
    elif arr.ndim == 3:
        channels = arr.shape[-1]
        mode_map = {1: 'L', 3: 'RGB', 4: 'RGBA'}
        if channels not in mode_map:
            raise ValueError(f"error channels:{channels}")
        mode = mode_map[channels]
    else:
        raise ValueError(f"error channels:{arr.ndim}")

    Image.fromarray(arr, mode=mode).save(filename)

def get_transformer():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def split_image(width, height,tile_size):
    """将图像分割成指定大小的小块"""
    top_list = []
    left_list = []
    for i in range(0, height, tile_size):
        top_list.append(i)
    if height % tile_size != 0:
        top_list.pop()
        top_list.append(height - tile_size)
    for j in range(0, width, tile_size):
        left_list.append(j)
    if width % tile_size != 0:
        left_list.pop()
        left_list.append(width - tile_size)
    positions = []
    for top in top_list:
        for left in left_list:
            bottom = min(top + tile_size - 1, height)
            right = min(left + tile_size - 1, width)
            positions.append([left, top, right, bottom])

    return positions

class ImageBlocks:
    def __init__(self, width, height, positions):
        """
        初始化时记录图像的尺寸、分块尺寸以及计算每个块的位置
        """
        self.width = width
        self.height = height
        self.positions = positions
        self.blocks = {}  # 使用字典存储：键为位置元组(left, top, right, bottom)，值为预测结果张量
        self.image = None  # 最终拼接后的完整预测图像

    def add_block(self, pos, block):
        """
        将预测得到的块添加到blocks字典中
        :param pos: [left, top, right, bottom]
        :param block: 预测结果张量，shape: (C, h, w)
        """
        self.blocks[tuple(pos)] = block

    def merge_blocks(self):
        """
        将所有块合并为完整的预测图像，处理重叠区域时取平均
        返回值为一个numpy数组，shape为 (H, W, C)
        """
        result = torch.zeros((3, self.height, self.width), dtype=torch.float32)

        # 对于每个块，将对应区域累加
        for pos, block in reversed(self.blocks.items()):
            left, top, right, bottom = pos
            block = block.squeeze(0)
            # 根据crop得到的块大小，可能小于tile_size
            h = block.shape[1]
            w = block.shape[2]

            result[:, top:(top+h), left:(left+w)] = block[:, :h, :w]


        # 保存最终拼接结果，并转换成(H, W, C)
        self.image = result
        return self.image
