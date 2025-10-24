import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1, conv3x3
from typing import Type, Union
from models.networks import ConvWithActivation, get_pad, DeConvWithActivation, DoubleConv, Up, PPM, FFP, OutConv, \
    adjust_size, ELA


class ResNet_UNet(nn.Module):
    def get_name(self):
        return "Release_lightly_extra"

    def return_num(self):
        return 5

    def __init__(self, n_channels=3, n_classes=3):
        super(ResNet_UNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 16
        self.base_width = 64
        self.dilation = 1
        self.groups = 1
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.block = BasicBlock

        self.inc = (DoubleConv(n_channels, 16))
        # self.down1 = (Down(64, 128))
        self.down1 = self._make_layer(self.block, 32, 3, stride=2)
        # self.down2 = (Down(128, 256))
        self.down2 = self._make_layer(self.block, 64, 4, stride=2)
        # self.down3 = (Down(256, 512))
        self.down3 = self._make_layer(self.block, 128, 6, stride=2)
        # self.down4 = (Down(512, 1024))
        self.down4 = self._make_layer(self.block, 256, 3, stride=2)

        self.up1 = (Up(256, 128))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(64, 32))
        self.up4 = (Up(32, 16))
        self.ffp = FFP()

        self.fusion = ConvWithActivation(16, 32, kernel_size=3, stride=1, padding=1)
        self.outc = (OutConv(32, n_classes))

        self.xo1 = DoubleConv(64, 3)
        self.xo2 = DoubleConv(32, 3)
        self.mask1 = Up(64, 32)
        self.mask2 = Up(32, 32)
        self.mask3 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)

        x1 = self.inc(x)  # shape0

        x2 = self.down1(x1)  # shape1

        x3 = self.down2(x2)  # shape2

        x4 = self.down3(x3)

        x5 = self.down4(x4)


        x = self.up1(x5, x4)


        x = self.up2(x, x3)
        xo1 = self.xo1(x)

        mm = self.mask1(x, x2)


        mm = self.mask2(mm, x1)
        mm = self.mask3(mm)


        x = self.up3(x, x2)
        xo2 = self.xo2(x)

        x = self.up4(x, x1)


        x = self.fusion(x)  # shape0

        x = self.outc(x)
        xo1 = F.interpolate(xo1, scale_factor=2, mode='bilinear', align_corners=False)
        xo2 = F.interpolate(xo2, scale_factor=2, mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        mm = F.interpolate(mm, scale_factor=2, mode='nearest')
        return xo1, xo2, x,x, mm

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

        return nn.Sequential(*layers, ELA(planes))



# truedata Batch: [96/96] Epoch Avg Loss: 0.7753 L1 Loss: 0.0193 MSE Loss: 0.0049 PSNR: 25.6812 SSIM: 0.9654 Avg time/batch: 0.449s Elapsed: 0:00:43 ETA: 0:00:00  cost time: 43.120593 s
if __name__ == '__main__':
    import random

    w = 512  # int(random.Random().random() * 128)+128
    h = 512  # int(random.Random().random() * 128)+128
    print(h, w)
    model = ResNet_UNet(n_channels=3, n_classes=3)

    # x = torch.randn(2, 3, h, w)  # Example input
    # xo1, xo2, xo3, x, mm = model(x)
    # print(xo1.shape,xo2.shape,x.shape,mm.shape)  # Should be (1, n_classes, 388, 388)

    model.eval()
    from torchinfo import summary

    summary(model, input_size=(2, 3, 256, 256), device="cpu")

