import os
from tools.utils import batch_images_build
from tools.dir_setting import Dir
if __name__ == '__main__':
    dir = Dir()
    dir.validate_paths()  # 验证路径有效性
    # 直接使用动态路径
    batch_images_build(dir.images_dir, dir.erased_binary_dir, dir.rebuild_dir)