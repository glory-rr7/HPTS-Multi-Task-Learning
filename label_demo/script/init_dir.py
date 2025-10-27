import os

'''
    运行会检查几个必要目录是否存在，缺失目录会创建
    如果存在几个目录，会清空目录下的文件
'''
class Dir:
    def __init__(self):
        # 获取当前脚本所在目录（label_demo/script）
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # 计算相对路径（顶层目录为label_demo和truedata的父目录）
        top_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

        # 定义label_demo下的demo目录为基准路径
        demo_dir = os.path.join(top_dir, 'label_demo', 'demo')

        # 加载和保存路径均基于demo目录
        self.images_dir = os.path.join(demo_dir, 'images')  # 原图加载路径（demo/images）
        self.images_binary_dir = os.path.join(demo_dir, 'images_binary')  # 二值化结果保存路径
        self.erased_binary_dir = os.path.join(demo_dir, 'erased_binary')  # 擦除结果保存路径
        self.rebuild_dir = os.path.join(demo_dir, 'rebuild')  # 重建结果保存路径
        self.erased_dir = os.path.join(demo_dir, 'erased')
        self.mask_dir = os.path.join(demo_dir, 'mask')
        self.overlap_mask_dir = os.path.join(demo_dir, 'overlap_mask')

    def validate_paths(self):
        # 验证路径存在性并清空现有目录文件
        for attr in ['images_dir', 'images_binary_dir', 'erased_binary_dir', 'rebuild_dir', 'erased_dir', 'mask_dir', 'overlap_mask_dir']:
            path = getattr(self, attr)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                print(f"已自动创建缺失目录: {path}")
            else:
                # 清空目录下的文件（保留目录）
                for filename in os.listdir(path):
                    file_path = os.path.join(path, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"已删除文件: {file_path}")

if __name__ == '__main__':
    dir = Dir()
    dir.validate_paths()  # 初始化时验证路径并清空文件