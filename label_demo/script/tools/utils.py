import os
import cv2
import numpy as np


def binarize_with_threshold(image, threshold, region):
    '''
    对图像的部分区域进行二值化
    Args:
        image: 输入图像（numpy数组）
        threshold: 二值化阈值（-1表示使用OTSU自动阈值）
        region: 要二值化的区域，格式为 (x, y, width, height)
                其中(x,y)为左上角坐标，width为宽度，height为高度
    Returns:
        二值化后的区域图像（numpy数组）
    '''
    # 解析区域参数（确保参数顺序正确）
    x, y, w, h = region
    # 确保区域不超出图像范围（防止索引越界）
    h_max = image.shape[0] - y
    w_max = image.shape[1] - x
    h = min(h, h_max)
    w = min(w, w_max)
    if h <= 0 or w <= 0:
        raise ValueError(f"区域超出图像范围或尺寸无效：region={region}, 图像尺寸={image.shape}")

    region_image = image[y:y + h, x:x + w]

    if threshold == -1:
        _, binary_image = cv2.threshold(region_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary_image = cv2.threshold(region_image, threshold, 255, cv2.THRESH_BINARY)

    return binary_image


def binarize_single_image(image_path, thresholds, output_path, file_name):
    try:
        # 规范路径拼接
        full_path = os.path.join(image_path, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"文件不存在: {full_path}")
        
        # 使用OpenCV支持中文路径的方式读取图像
        img_array = np.fromfile(full_path, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"无法解码图像: {full_path}")
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_path, exist_ok=True)
        
        # 移除冗余的cv2.imread调用，使用已解码的image变量
        # 验证阈值列表长度
        if len(thresholds) != 4:
            print(f"错误：thresholds必须包含4个阈值，实际为{len(thresholds)}个")
            return False
        
        # 获取图像尺寸
        height, width = image.shape

        # 定义四个区域（x, y, width, height），确保边缘不溢出
        regions = [
            (0, 0, width // 2, height // 2),  # 左上
            (width // 2, 0, width - width // 2, height // 2),  # 右上（避免宽度奇数时的溢出）
            (0, height // 2, width // 2, height - height // 2),  # 左下
            (width // 2, height // 2, width - width // 2, height - height // 2)  # 右下
        ]

        # 分区域二值化
        processed_regions = []
        for i, region in enumerate(regions):
            try:
                processed_region = binarize_with_threshold(image, thresholds[i], region)
                processed_regions.append(processed_region)
            except Exception as e:
                print(f"错误：处理区域{i}失败 - {str(e)}")
                return False

        # 拼接区域（确保尺寸匹配）
        try:
            top_half = np.hstack([processed_regions[0], processed_regions[1]])
            bottom_half = np.hstack([processed_regions[2], processed_regions[3]])
            final_image = np.vstack([top_half, bottom_half])
        except ValueError as e:
            print(f"错误：区域拼接失败（尺寸不匹配） - {str(e)}")
            return False

        # 输出路径处理（替换扩展名为.png）
        name, _ = os.path.splitext(file_name)
        output_file = os.path.join(output_path, f"{name}.png")

        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)

        # 保存结果
        if not cv2.imwrite(output_file, final_image):
            print(f"错误：无法保存图像 - {output_file}")
            return False

        print(f"成功：二值化完成 - {output_file}")
        return True

    except Exception as e:
        print(f"处理图像时发生未知错误 - {str(e)}")
        return False


def batch_images_build(images_dir, erased_binary_dir, rebuild_dir):
    '''
    批量重建图像：结合原始彩色图像和擦除二值化图像，保留有效区域
    '''
    os.makedirs(rebuild_dir, exist_ok=True)

    # 过滤仅处理.jpg文件（避免非图像文件干扰）
    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f))
           and f.lower().endswith('.jpg')
    ]

    for image_file in image_files:
        try:
            # 构建路径
            color_path = os.path.join(images_dir, image_file)
            erased_path = os.path.join(erased_binary_dir, image_file.replace('jpg', 'png'))

            # 检查擦除文件是否存在
            if not os.path.exists(erased_path):
                print(f"警告：擦除文件不存在，跳过 - {erased_path}")
                continue

            # 读取图像并检查
            color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
            if color_img is None:
                print(f"错误：无法读取彩色图像 - {color_path}")
                continue

            binary_img = cv2.imread(erased_path, cv2.IMREAD_GRAYSCALE)
            if binary_img is None:
                print(f"错误：无法读取二值化图像 - {erased_path}")
                continue

            # 检查图像尺寸是否匹配
            if color_img.shape[:2] != binary_img.shape[:2]:
                print(f"错误：图像尺寸不匹配 - {image_file}（彩色：{color_img.shape[:2]}，二值化：{binary_img.shape[:2]}）")
                continue

            # 创建白色背景
            output_img = np.full_like(color_img, 255, dtype=np.uint8)

            # 复制黑色区域（保留原始像素）
            black_pixels = (binary_img == 0)  # 二值化图像中黑色区域为有效区域
            output_img[black_pixels] = color_img[black_pixels]

            # 保存结果
            output_path = os.path.join(rebuild_dir, image_file)
            if not cv2.imwrite(output_path, output_img):
                print(f"错误：无法保存重建图像 - {output_path}")
                continue

            print(f"成功：重建图像 - {output_path}")

        except Exception as e:
            print(f"处理{image_file}时出错 - {str(e)}")
            continue


def batch_images_mask(images_binary_dir, erased_binary_dir, overlap_mask_dir, output_dir):
    '''
    生成三色掩码图像：红色=原始二值化区域，绿色=擦除后区域，蓝色=重叠区域
    '''
    os.makedirs(output_dir, exist_ok=True)

    # 过滤仅处理.png文件（假设掩码图像为PNG格式）
    def get_image_files(dir_path):
        return [
            f for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
               and f.lower().endswith('.png')
        ]

    # 获取三个目录的图像文件
    img_files = set(get_image_files(images_binary_dir))
    erased_files = set(get_image_files(erased_binary_dir))
    overlap_files = set(get_image_files(overlap_mask_dir))
    common_files = img_files & erased_files & overlap_files  # 取交集
    for filename in common_files:
        try:
            # 构建路径
            img_path = os.path.join(images_binary_dir, filename)
            erased_path = os.path.join(erased_binary_dir, filename)
            overlap_path = os.path.join(overlap_mask_dir, filename)

            # 读取图像并检查
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"错误：无法读取原始二值化图像 - {img_path}")
                continue

            erased = cv2.imread(erased_path, cv2.IMREAD_GRAYSCALE)
            if erased is None:
                print(f"错误：无法读取擦除后图像 - {erased_path}")
                continue

            # 读取带alpha通道的重叠掩码（检查通道数）
            overlap_color = cv2.imread(overlap_path, cv2.IMREAD_UNCHANGED)
            if overlap_color is None:
                print(f"错误：无法读取重叠掩码 - {overlap_path}")
                continue
            if overlap_color.ndim != 3 or overlap_color.shape[2] < 4:
                print(f"错误：重叠掩码必须包含alpha通道（4通道） - {overlap_path}")
                continue

            # 检查图像尺寸是否一致
            if img.shape != erased.shape or img.shape != overlap_color.shape[:2]:
                print(f"错误：图像尺寸不匹配 - {filename}")
                continue

            # 二值化（确保黑白分明）
            _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            _, erased_bin = cv2.threshold(erased, 127, 255, cv2.THRESH_BINARY)

            # 创建白色背景的彩色掩码（RGB通道）
            mask = np.full((img.shape[0], img.shape[1], 3), 255, dtype=np.uint8)

            # 原始二值化区域（黑色→红色）
            mask[img_bin == 0] = [0, 0, 255]  # BGR格式（OpenCV默认），红色为(0,0,255)

            # 擦除后区域（黑色→绿色）
            mask[erased_bin == 0] = [0, 255, 0]  # 绿色

            # 重叠区域（alpha通道非0→蓝色）
            alpha_channel = overlap_color[:, :, 3]  # 提取alpha通道
            mask[alpha_channel > 0] = [255, 0, 0]  # 蓝色

            # 保存结果
            output_path = os.path.join(output_dir, filename)
            if not cv2.imwrite(output_path, mask):
                print(f"错误：无法保存掩码图像 - {output_path}")
                continue

            print(f"成功：生成掩码图像 - {output_path}")

        except Exception as e:
            print(f"处理{filename}时出错 - {str(e)}")
            continue
