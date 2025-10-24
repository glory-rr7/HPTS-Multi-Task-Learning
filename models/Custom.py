import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck,BasicBlock,conv1x1,conv3x3
from typing import  Type, Union
from models.networks import ConvWithActivation, Refinement, DoubleConv,Up,PPM,FFP,OutConv,adjust_size
from models.networks import ELA,CoordAtt,CBAM,ECA,SA,SE

# base: 32  # 参数量,默认32，必须是8的倍数，建议8--64，可参考满血版位64，轻量版32
# refinement: True  # 精修复模块，与骨干网络串行的网络模块，占用额外显存较小，但会增加耗时
# ffp: True  # 细化特征模块，与骨干网络并行的网络模块，占用额外显存较大，额外耗时较小
# ppm: True  # 在最深处使用金字塔池化模块，可以捕捉大范围的图像特征
# down_sample: 1  # 下采样倍数，默认1表示不进行下采样，建议0.5或者0.75
# am: ela  # 注意力机制模块，经过实验证明，通道/空间注意力机制模块更好

def get_am( am):
    if am == 'ela':#4,531,596
        return ELA
    elif am == 'ca':#4,535,772
        return CoordAtt
    elif am == 'cbam':#4,532,807
        return CBAM
    elif am == 'eca':#4,529,448
        return ECA
    elif am == 'sa':#4,529,526
        return SA
    elif am == 'se':#4,532,156
        return SE

class ResNet_UNet(nn.Module):
    def get_name(self):
        return "Custom"

    def return_num(self):
        return 5


    def __init__(self, n_channels=3, n_classes=3, base=24, refinement=True, ffp=True, ppm=True, down_sample=1.0, am='ela'):
        super(ResNet_UNet, self).__init__()
        print("base:", base)
        print("refinement:", refinement)
        print("ffp:", ffp)
        print("ppm:", ppm)
        print("down_sample:", down_sample)
        print("am:", am)

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = base//2
        self.base_width = 64
        self.dilation = 1
        self.groups = 1
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.block = BasicBlock
        AM = get_am(am)
        self.inc = (DoubleConv(n_channels, self.inplanes))
        self.down1 = self._make_layer(self.block, base, 3,stride=2, AM=AM)
        self.down2 = self._make_layer(self.block, base*2, 4, stride=2,AM=AM)
        self.down3 = self._make_layer(self.block, base*4, 6, stride=2,AM=AM)
        down4 = self._make_layer(self.block, base*8, 3, stride=2,AM=AM)

        if ppm==True:
            self.down4 = nn.Sequential(
                down4,PPM(base*8)
            )
        else:
            self.down4 = down4


        self.up1 = Up(base*8, base*4)
        self.up2 = Up(base*4, base*2 )
        self.up3 = Up(base*2, base)
        self.up4 = Up(base, base//2)


        if ffp == True:
            self.ffp = FFP()
            self.fusion = ConvWithActivation(base // 2 + 32, base // 2, kernel_size=3, stride=1, padding=1)
        else:
            self.ffp = None
            self.fusion = ConvWithActivation(base // 2, base // 2, kernel_size=3, stride=1, padding=1)
        self.outc = (OutConv(base//2, n_classes))

        self.xo1 = DoubleConv(base*2, 3)
        self.xo2 = DoubleConv(base, 3)
        self.mask1 = Up(base*2, base)
        self.mask2 = Up(base, base//2)
        self.mask3 = nn.Conv2d(base//2, 4,kernel_size=3,padding=1)

        if refinement == True:
            self.refinement = Refinement(base)
        else:
            self.refinement = None

        self.down_sample = down_sample
    def forward(self, x):
        if self.down_sample != 1:
            x = F.interpolate(x, scale_factor=self.down_sample, mode='bilinear', align_corners=True)
        identity = x

        shape0 = x.shape
        x1 = self.inc(x)#shape0
        con_x1 = x1
        #print('x1:', x1.shape)
        x2 = self.down1(x1)#shape1
        shape1 = x2.shape
        con_x2 = x2
        x3 = self.down2(x2)#shape2

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        #print(x4.shape,x5.shape)
        x = self.up1(x5, x4)
        #print(x.shape)

        x = self.up2(x, x3)
        xo1 = self.xo1(x)
        #print(x.shape,x2.shape)
        mm = self.mask1(x, x2)

        mm = self.mask2(mm, x1)
        mm = self.mask3(mm)
        #print(x.shape)

        x = self.up3(x, x2)
        xo2 = self.xo2(x)
        #print(x.shape)
        x = self.up4(x, x1)

        if self.ffp is not None:
            ffp = self.ffp(identity)
            x = self.fusion(torch.cat([x,ffp], dim=1))  # shape0
            del ffp
        else:
            x = self.fusion(x)

        x_o_unet = self.outc(x)

        if self.refinement is not None:
            x = self.refinement(x, identity,con_x1,con_x2,shape0,shape1)
        else:
            x = x_o_unet
        if self.down_sample != 1:
            factor = 1 / self.down_sample
            xo1 = F.interpolate(xo1, scale_factor=factor, mode='bilinear', align_corners=False)
            xo2 = F.interpolate(xo2, scale_factor=factor, mode='bilinear', align_corners=False)
            x_o_unet = F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)
            x = F.interpolate(x, scale_factor=factor, mode='bilinear', align_corners=False)
            mm = F.interpolate(mm, scale_factor=factor, mode='nearest')
        return xo1, xo2, x_o_unet, x, mm

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
            AM = ELA
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

        return nn.Sequential(*layers,AM(planes))


if __name__ == '__main__':
    import random

    w = 256#int(random.Random().random() * 128)+128
    h = 256#int(random.Random().random() * 128)+128
    print(h, w)
    model = ResNet_UNet(n_channels=3, n_classes=3, base=24,ppm=True,refinement=True,down_sample=1.0,ffp=True,am='ela')
    #print(model)
    x = torch.randn(2, 3, h, w)  # Example input
    xo1, xo2, xo3, x, mm = model(x)
    # print(xo1.shape,xo2.shape,x.shape,mm.shape)  # Should be (1, n_classes, 388, 388)
    # from torchinfo import summary
    # summary(model, input_size=(2, 3, 256, 256),device="cpu")
