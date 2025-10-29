import numpy as np
import cv2
from rknn.api import RKNN
import torch
import os



# 模型文件路径
model_path = '../v2_1_bg_500.pt'

# 检查模型文件是否存在
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    exit(1)

# 定义输入形状
input_size_list = [[1, 3, 512, 512]]

# 创建 RKNN 对象
rknn = RKNN(verbose=True)

# 配置预处理参数
print('--> Config model')
rknn.config(target_platform="rk3576",mean_values=[255*0.485, 255*0.456, 255*0.406], std_values=[255*0.229, 255*0.224, 255*0.225])
#rknn.config(target_platform="rk3576", mean_values=[0.485, 0.456,0.406],std_values=[0.229,0.224, 0.225])
print('done')

# 加载 PyTorch 模型
print('--> Loading model')
ret = rknn.load_pytorch(model=model_path, input_size_list=input_size_list)
if ret != 0:
    print('Load model failed!')
    exit(ret)
print('done')

# 编译模型
print('--> Building model')
#ret = rknn.build(do_quantization=True, dataset='./dataset.txt',rknn_batch_size=2)
ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
if ret != 0:
    print('Build model failed!')
    exit(ret)
print('done')

# 导出 RKNN 模型
print('--> Export rknn model')
ret = rknn.export_rknn('./v2_1_bg_500_ext.rknn')
if ret != 0:
    print('Export rknn model failed!')
    exit(ret)
print('done')

# 初始化运行时环境
print('--> Init runtime environment')

rknn.config(target_platform="rk3576")
ret = rknn.init_runtime()
if ret != 0:
    print('Init runtime environment failed!')
    exit(ret)
print('done')

# # 推理测试
# print('--> Running model')
# img = cv2.imread('./test_image.jpg')  # 替换为实际测试图片路径
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (224, 224))  # 调整图片大小匹配模型输入
# outputs = rknn.inference(inputs=[img])
# np.save('./output.npy', outputs[0])  # 保存输出
# show_outputs(softmax(np.array(outputs[0][0])))
# print('done')




















# 释放 RKNN 资源
rknn.release()
