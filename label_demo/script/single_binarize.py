from tools.utils import binarize_single_image
from tools.dir_setting import Dir
import os
# 该脚本用于将图像分割成四个区域，并分别进行二值化处理，生成四个区域的二值图像
# 实际上就是讲一张彩色图像转变成黑白图像，方便后续操作
# 至于为什么要分成不同区域，是因为有的图像存在部分区域阴影严重，如果全局二值化，阴影造成的较低阈值会使文字处理后极其不清晰
# 为什么要单张二值化，而不是直接批处理，是因为每张图片的阈值都不一样，为了避免大量噪声和阴影，只能单张处理

if __name__ == '__main__':
    dir = Dir()
    file_name = 'cis_bmp_20250325_203746.jpg'  # 图像文件名

    thresholds = [100, 120,  # 分别代表了左上，右上，左下，右下四个区域的阈值
                  100, 100]  # 阈值越低，文字保留越不完整，噪音越小
    # 阈值越高，文字保留越完整，但同时噪音越大
    # 阈值建议85-125之间，若阴影很严重，则可以小于85；若文字保留不完整，则可以大于125

    #-----------------------------------原图二值化-----------------------------------#

    # images中的图片名应当是jpg结尾的，这里用来确保文件后缀是jpg
    images_file_name = file_name.replace('png','jpg')
    binarize_single_image(dir.images_dir, thresholds, dir.images_binary_dir, images_file_name)
    # -----------------------------------------------------------------------------#



    #-------------------------------手写字擦除图像二值化-------------------------------#

    # erased中的图片名应当是png结尾的，这里用来确保文件后缀是png
    erased_file_name = file_name.replace('jpg','png')
    binarize_single_image(dir.erased_dir, thresholds, dir.erased_binary_dir, erased_file_name)
    # -----------------------------------------------------------------------------#
    # 在main函数开头添加路径验证
    dir = Dir()
    dir.validate_paths()  # 新增验证
    
    # 修改文件处理逻辑
    if not os.path.exists(os.path.join(dir.images_dir, file_name)):
        raise FileNotFoundError(f"源图像不存在: {os.path.join(dir.images_dir, file_name)}")
    
    # 使用更安全的扩展名处理方式
    base_name = os.path.splitext(file_name)[0]
    erased_file_name = f"{base_name}.png"