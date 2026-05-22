import torch
import torch.nn as nn

from models.model_output import make_model_output
from models.networks import ConvWithActivation, DoubleConv, ELA, FFP, OutConv, Refinement


class SegUp(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y < 0 or diff_x < 0:
            x = x[:, :, :skip.size(2), :skip.size(3)]
        else:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        return self.conv(torch.cat([skip, x], dim=1))


class Layer(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_d = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1_d = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.conv2_d = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.bn2_d = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def IEforward(self, x):
        self.conv1_d.weight.data = self.conv1.weight.data
        self.bn1_d.weight.data = self.bn1.weight.data
        self.bn1_d.bias.data = self.bn1.bias.data
        self.conv2_d.weight.data = self.conv2.weight.data
        self.bn2_d.weight.data = self.bn2.weight.data
        self.bn2_d.bias.data = self.bn2.bias.data

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return self.relu(out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if len(self.shortcut) > 0:
            out = out + self.shortcut(x)
            return self.relu(out)

        x_plus = self.relu(out + x)
        x_plus.requires_grad_(True)
        with torch.set_grad_enabled(True):
            f_x_plus = self.IEforward(x_plus)
            ie_loss = torch.norm(x_plus - x - f_x_plus) ** 2
            grad = torch.autograd.grad(ie_loss, x_plus, retain_graph=True, create_graph=False)
            x_plus = x_plus - 0.05 * grad[0]
        return x_plus


class BasicBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, stride):
        super().__init__()
        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            layer_in = in_channels if layer_idx == 0 else out_channels
            layer_stride = stride if layer_idx == 0 else 1
            self.layers.append(Layer(layer_in, out_channels, layer_stride))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ResNet_UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, mask_classes=4, model_name="IEResNet"):
        super().__init__()
        self.model_name = model_name
        self.mask_classes = mask_classes

        num_blocks = [2, 2, 2, 2]
        self.refine_skip1 = DoubleConv(n_channels, 64)
        self.refine_skip2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_stage(BasicBlock, 64, 64, num_blocks[0], 1)
        self.layer2 = self._make_stage(BasicBlock, 64, 128, num_blocks[1], 2)
        self.layer3 = self._make_stage(BasicBlock, 128, 256, num_blocks[2], 2)
        self.layer4 = self._make_stage(BasicBlock, 256, 512, num_blocks[3], 2)

        self.ela_stem = ELA(64)
        self.ela1 = ELA(64)
        self.ela2 = ELA(128)
        self.ela3 = ELA(256)
        self.ela4 = ELA(512)

        self.up1 = SegUp(512, 256, 256)
        self.up2 = SegUp(256, 128, 128)
        self.up3 = SegUp(128, 64, 64)
        self.up4 = SegUp(64, 64, 64)
        self.up5 = SegUp(64, 64, 64)

        self.xo1 = DoubleConv(64, n_classes)
        self.xo2 = DoubleConv(64, n_classes)
        self.mask1 = SegUp(64, 64, 64)
        self.mask2 = SegUp(64, 64, 64)
        self.mask3 = nn.Conv2d(64, mask_classes, kernel_size=3, padding=1)

        self.ffp = FFP()
        self.fusion = ConvWithActivation(96, 64, kernel_size=3, stride=1, padding=1)
        self.outc = OutConv(64, n_classes)
        self.refinement = Refinement(128)

        self._init_weights()

    def get_name(self):
        return self.model_name

    def return_num(self):
        return 5

    def _make_stage(self, block, in_channels, out_channels, num_layers, stride):
        return block(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        identity = x
        shape0 = x.shape

        con_x1 = self.refine_skip1(x)
        stem = self.conv1(x)
        stem = self.bn1(stem)
        stem = self.relu(stem)
        stem = self.ela_stem(stem)
        con_x2 = self.refine_skip2(stem)
        shape1 = stem.shape

        x = self.maxpool(stem)
        x1 = self.ela1(self.layer1(x))
        x2 = self.ela2(self.layer2(x1))
        x3 = self.ela3(self.layer3(x2))
        x4 = self.ela4(self.layer4(x3))

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        xo1 = self.xo1(x)

        mm = self.mask1(x, stem)
        mm = self.mask2(mm, con_x1)
        mm = self.mask3(mm)

        x = self.up4(x, stem)
        xo2 = self.xo2(x)
        x = self.up5(x, con_x1)
        x = self.fusion(torch.cat([x, self.ffp(identity)], dim=1))
        x_o_unet = self.outc(x)
        output = self.refinement(x, identity, con_x1, con_x2, shape0, shape1)

        return make_model_output(
            x1=xo1,
            x2=xo2,
            x3=x_o_unet,
            output=output,
            mask_logits=mm,
        )
