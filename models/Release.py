import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck,BasicBlock,conv1x1,conv3x3
from typing import  Type, Union
from networks import ConvWithActivation, get_pad, DeConvWithActivation,DoubleConv,Up,PPM,FFP,OutConv,adjust_size, ELA



class ResNet_UNet(nn.Module):
    def get_name(self):
        return "Release"

    def return_num(self):
        return 5

    def __init__(self, n_channels=3, n_classes=3):
        super(ResNet_UNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 32
        self.base_width = 64
        self.dilation = 1
        self.groups = 1
        self.n_channels = n_channels 
        self.n_classes = n_classes
        self.block = BasicBlock

        self.inc = (DoubleConv(n_channels, 32))
        #self.down1 = (Down(64, 128))
        self.down1 = self._make_layer(self.block, 64, 3,stride=2)
        #self.down2 = (Down(128, 256))
        self.down2 = self._make_layer(self.block, 128, 4, stride=2)
        #self.down3 = (Down(256, 512))
        self.down3 = self._make_layer(self.block, 256, 6, stride=2)
        #self.down4 = (Down(512, 1024))
        self.down4 = self._make_layer(self.block, 512, 3, stride=2)
        self.ppm = PPM(512)
        self.up1 = (Up(512, 256 ))
        self.up2 = (Up(256, 128 ))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(64, 32))
        self.ffp = FFP()

        self.fusion = ConvWithActivation(64, 32, kernel_size=3, stride=1, padding=1)
        self.outc = (OutConv(32, n_classes))

        self.xo1 = DoubleConv(128, 3)
        self.xo2 = DoubleConv(64, 3)
        self.mask1 = Up(128, 64)
        self.mask2 = Up(64, 32)
        self.mask3 = nn.Conv2d(32, 4,kernel_size=3,padding=1)

        ##### Refine sub-network ######
        n_in_channel = 3
        cnum = 32
        ####downsapmle
        self.coarse_conva = ConvWithActivation(n_in_channel+32, cnum, kernel_size=5, stride=1, padding=2)
        self.coarse_convb = ConvWithActivation(cnum, 2 * cnum, kernel_size=4, stride=2, padding=1)
        self.coarse_convc = ConvWithActivation(2 * cnum, 2 * cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_convd = ConvWithActivation(2 * cnum, 4 * cnum, kernel_size=4, stride=2, padding=1)
        self.coarse_conve = ConvWithActivation(4 * cnum, 4 * cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_convf = ConvWithActivation(4 * cnum, 4 * cnum, kernel_size=3, stride=1, padding=1)
        ### astrous
        self.astrous_net = nn.Sequential(
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
        )
        ###astrous
        ### upsample
        self.coarse_convk = ConvWithActivation(4 * cnum, 4 * cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_convl = ConvWithActivation(4 * cnum, 4 * cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_deconva = DeConvWithActivation(4 * cnum * 3, 2 * cnum, kernel_size=3, padding=1, stride=2)
        self.coarse_convm = ConvWithActivation(2 * cnum, 2 * cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_deconvb = DeConvWithActivation(2 * cnum * 3, cnum, kernel_size=3, padding=1, stride=2)
        self.coarse_convn = nn.Sequential(
            ConvWithActivation(cnum, cnum // 2, kernel_size=3, stride=1, padding=1),
            # Self_Attn(cnum//2, 'relu'),
            ConvWithActivation(cnum // 2, 3, kernel_size=3, stride=1, padding=1, activation=None),
        )
        self.c1 = nn.Conv2d(32, 64, kernel_size=1)
        self.c2 = nn.Conv2d(64, 128, kernel_size=1)
        ##### Refine network ######

    def forward(self, x):
        identity = x

        shape0 = x.shape
        x1 = self.inc(x)#shape0
        con_x1 = x1

        x2 = self.down1(x1)#shape1
        shape1 = x2.shape
        con_x2 = x2
        x3 = self.down2(x2)#shape2

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        #print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape)
        x5 = self.ppm(x5)
        #print(x4.shape,x5.shape)
        x = self.up1(x5, x4)
        #print(x.shape)

        x = self.up2(x, x3)
        xo1 = self.xo1(x)
        mm = self.mask1(x, x2)
        mm = self.mask2(mm, x1)
        mm = self.mask3(mm)
        #print(x.shape)


        x = self.up3(x, x2)
        xo2 = self.xo2(x)
        #print(x.shape)
        x = self.up4(x, x1)
        #print(x.shape)
        ffp = self.ffp(identity)
        #print("down")

        x = self.fusion(torch.cat([x,ffp], dim=1))  # shape0
        del ffp
        x_o_unet = self.outc(x)

        ###refine sub-network

        x = self.coarse_conva(torch.cat([x,identity],dim=1))  # shape0

        x = self.coarse_convb(x)  # shape1

        x = self.coarse_convc(x)  # shape1

        x_c1 = x  ###concate feature1
        x = self.coarse_convd(x)  # shape2

        x = self.coarse_conve(x)  # shape2

        x = self.coarse_convf(x)  # shape2

        x_c2 = x  ###concate feature2
        x = self.astrous_net(x)  # shape2

        x = self.coarse_convk(x)  # shape2

        x = self.coarse_convl(x)  # shape2


        temp = self.c2(con_x2)
        temp = F.max_pool2d(temp, kernel_size=2, stride=2)
        x, x_c2, temp = adjust_size([x, x_c2, temp], shape1[2], shape1[3])
        x = self.coarse_deconva(torch.cat([x, x_c2, temp], dim=1))
        x = self.coarse_convm(x)  # shape1
        [x] = adjust_size([x], shape1[2] * 2, shape1[3] * 2)
        del x_c2
        temp = self.c1(con_x1)
        temp = F.max_pool2d(temp, kernel_size=2, stride=2)
        x, x_c1, temp = adjust_size([x, x_c1, temp], shape0[2], shape0[3])
        x = self.coarse_deconvb(torch.cat([x, x_c1, temp], dim=1))
        x = self.coarse_convn(x)  # shape0
        [x] = adjust_size([x], shape0[2] * 2, shape0[3] * 2)
        return xo1, xo2, x_o_unet, x, mm

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers,ELA(planes))


if __name__ == '__main__':
    import random

    # w = 512#int(random.Random().random() * 128)+128
    # h = 512#int(random.Random().random() * 128)+128
    # print(h, w)
    # model = ResNet_UNet(n_channels=3, n_classes=3)
    # x = torch.randn(2, 3, h, w)  # Example input
    # model.forward(x)
    #xo1, xo2, xo3, x, mm = model(x)
    #print(xo1.shape,xo2.shape,x.shape,mm.shape)  # Should be (1, n_classes, 388, 388)
    # from torchinfo import summary
    # summary(model, input_size=(2, 3, 256, 256),device="cpu")

    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np

    # 加载图像
    image_path1 = "../datasets/demo/images/4.png"  # 第一张图像路径
    image_path2 = "../datasets/demo/images/5.png"  # 第二张图像路径
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # 定义转换
    transform = transforms.ToTensor()

    # 转换图像为张量
    tensor_image1 = transform(image1)
    tensor_image2 = transform(image2)

    # 将两张图像堆叠在一起，形成一个批次 (batch_size=2)
    batch_image = torch.stack([tensor_image1, tensor_image2], dim=0)

    print(f"图像转换为张量的批次形状: {batch_image.shape}")  # 应该是 (2, 3, H, W)

    # 假设模型是一个 ResNet_UNet，接收一个批次输入
    model = ResNet_UNet(n_channels=3, n_classes=3)
    output = model.forward(batch_image)

    # 检查 output 的类型和内容
    print(f"Output 类型: {type(output)}")
    print(f"Output 内容: {output}")

    # 假设 output 是一个包含 5 个张量的元组，选择第一个张量作为输出
    output_image = output[4]  # 选择元组中的第一个张量进行可视化


    def visualize_and_save_rgb_image(tensor, save_path=None):
        """
        可视化 RGB 图像，并在可视化之后保存图像，假设张量形状为 (batch_size, 3, height, width)

        参数:
        - tensor: 形状为 (batch_size, 3, height, width) 的张量
        - save_path: 要保存图像的路径，如果为 None，则不保存
        """
        # 确保输入张量是4D (batch_size, channels, height, width)
        batch_size, channels, height, width = tensor.shape

        # 选择批次中的第一张图像 (如果是批次输入)
        image = tensor[0]  # 获取第 0 张图像，形状为 (channels, height, width)

        # 将 (C, H, W) 转为 (H, W, C)
        image = image.permute(1, 2, 0)  # 将 (C, H, W) 转换为 (H, W, C)

        # 如果张量的值在 [0, 1] 范围内，需要将其转换为 [0, 255]
        image = image.cpu().detach().numpy()  # 转为 numpy 数组
        image = (image * 255).astype('uint8')  # 转为 uint8 类型，并归一化到 [0, 255]

        # 显示图像
        plt.imshow(image)
        plt.axis('off')  # 关闭坐标轴
        plt.show()

        # 如果指定了保存路径，保存图像
        if save_path:
            # 将 NumPy 数组转换为 PIL 图像
            pil_image = Image.fromarray(image)
            pil_image.save(save_path)
            print(f"图像已保存至: {save_path}")


    # 可视化并保存批次中的第一张图像
    output_image_path = "../output_image5.png"  # 指定保存路径
    visualize_and_save_rgb_image(output_image, save_path=output_image_path)

