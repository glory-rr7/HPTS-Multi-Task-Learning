import os
from tools.utils import batch_images_mask
from tools.dir_setting import Dir
if __name__ == "__main__":

    dir = Dir()

    batch_images_mask(dir.images_binary_dir, dir.erased_binary_dir, dir.overlap_mask_dir, dir.mask_dir)

