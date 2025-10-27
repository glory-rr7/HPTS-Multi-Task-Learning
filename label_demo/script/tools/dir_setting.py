import os.path


class Dir():
    def __init__(self):
        # 恢复正确的根目录路径
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../label_demo/demo'))  # 改为label_demo下的demo目录
        self.images_dir = os.path.join(self.root_dir, 'images')                # 原图路径
        self.erased_dir = os.path.join(self.root_dir, 'erased')                # 擦除手写字图路径
        self.images_binary_dir = os.path.join(self.root_dir, "images_binary")  # 原图二值化图路径
        self.erased_binary_dir = os.path.join(self.root_dir, "erased_binary")  # 擦除手写字图的二值化图路径
        self.overlap_mask_dir = os.path.join(self.root_dir, "overlap_mask")    # 重叠区域路径
        self.mask_dir = os.path.join(self.root_dir, 'mask')                    # 蒙版路径
        self.rebuild_dir = os.path.join(self.root_dir, 'rebuild')              # 重建图路径

    def make_dirs(self):  # 创建目录
        dirs = [
            self.images_dir,
            self.erased_dir,
            self.images_binary_dir,
            self.erased_binary_dir,
            self.overlap_mask_dir,
            self.mask_dir,
            self.rebuild_dir
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
            print("dir has been created", d)

    def validate_paths(self):
        """验证目录存在性（移除文件验证）"""
        required_dirs = [
            (self.images_dir, '目录'),
            (self.erased_dir, '目录'),
            (self.images_binary_dir, '目录'),
            (self.erased_binary_dir, '目录'),
            (self.overlap_mask_dir, '目录'),
            (self.mask_dir, '目录'),
            (self.rebuild_dir, '目录')
        ]
        for path, type_name in required_dirs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{type_name}不存在: {path}")
