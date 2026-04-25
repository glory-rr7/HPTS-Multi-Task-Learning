import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)


class BuildOffsetTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, kernel_size=3, dilations=[1]):
        B, C, H, W = x.shape
        kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        patches_all = []
        for d in dilations:
            pad_h = (kh - 1) * d // 2
            pad_w = (kw - 1) * d // 2
            x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
            p = F.unfold(
                x_pad,
                kernel_size=(kh, kw),
                dilation=(d, d),
                padding=0,
                stride=1,
            )
            p = p.view(B, C, kh * kw, H, W).permute(0, 1, 3, 4, 2)
            patches_all.append(p)

        return torch.cat(patches_all, dim=4)


class LocalAttention(nn.Module):
    def __init__(self, in_channels=32, external_out_channels=16,
                 self_out_channels=16, feature_size=16):
        super(LocalAttention, self).__init__()

        if external_out_channels == 0 and self_out_channels == 0:
            raise ValueError("external_out_channels or self_out_channels must be greater than 0")

        self.in_channels = in_channels
        self.external_out_channels = external_out_channels
        self.self_out_channels = self_out_channels
        self.feature_size = feature_size
        self.out_channels = external_out_channels + self_out_channels

        self.use_external = external_out_channels > 0
        if self.use_external:
            self.k = nn.Parameter(torch.Tensor(in_channels, feature_size))
            self.v = nn.Parameter(torch.Tensor(feature_size, external_out_channels))
            nn.init.kaiming_uniform_(self.k, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.v, a=math.sqrt(5))

        self.use_self = self_out_channels > 0
        if self.use_self:
            self.offset_tensor = BuildOffsetTensor()
            self.v_project = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self_out_channels,
                kernel_size=1,
            )

    def forward(self, q, k=None):
        if k is None:
            k = q
        outputs = []
        attn_weight = []

        use_self_attn = (k is not None and q.shape == k.shape and self.use_self)
        use_external_attn = self.use_external

        if use_external_attn:
            q_permuted = q.permute(0, 2, 3, 1)
            attn = torch.matmul(q_permuted, self.k)
            attn = torch.softmax(attn, dim=-1)
            out_external = torch.matmul(attn, self.v)
            out_external = out_external.permute(0, 3, 1, 2)
            outputs.append(out_external)
            attn_weight.append(attn.permute(0, 3, 1, 2))

        if use_self_attn:
            v = self.v_project(k)

            k_shift = self.offset_tensor(k)
            v_shift = self.offset_tensor(v)

            q_permuted = q.permute(0, 2, 3, 1).unsqueeze(-2)
            k_shift_permuted = k_shift.permute(0, 2, 3, 1, 4)
            attn = torch.matmul(q_permuted, k_shift_permuted)
            attn = torch.softmax(attn, dim=-1)

            v_shift_permuted = v_shift.permute(0, 2, 3, 4, 1)
            attn_out = torch.matmul(attn, v_shift_permuted)
            attn_out = attn_out.squeeze(-2).permute(0, 3, 1, 2)
            outputs.append(attn_out)
            attn_weight.append(attn.squeeze(-2).permute(0, 3, 1, 2))

        if len(outputs) == 1:
            outputs = outputs[0]
            attn_weight = attn_weight[0]
        else:
            outputs = torch.cat(outputs, dim=1)
            attn_weight = torch.cat(attn_weight, dim=1)

        return {
            "x": outputs,
            "attn": attn_weight,
        }

class ConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(ConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

class DeConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(DeConvWithActivation, self).__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

def adjust_size(tensors, target_h, target_w):

    adj_h = target_h // 2 if target_h % 2 == 0 else (target_h + 1) // 2
    adj_w = target_w // 2 if target_w % 2 == 0 else (target_w + 1) // 2
    ret = []
    for tensor in tensors:
        current_h, current_w = tensor.shape[2], tensor.shape[3]

        # 调整高度
        if current_h < adj_h:
            pad_h = adj_h - current_h
            tensor = F.pad(tensor, (0, 0, 0, pad_h), mode='replicate')
        elif current_h > adj_h:
            end = current_h - adj_h
            tensor = tensor[:, :, :-end, :]

        # 调整宽度
        if current_w < adj_w:
            pad_w = adj_w - current_w
            tensor = F.pad(tensor, (0, pad_w, 0, 0), mode='replicate')
        elif current_w > adj_w:
            end = current_w - adj_w
            tensor = tensor[:, :, :, :-end]

        ret.append(tensor)

    return ret

class FFPStage(nn.Module):
    """单个FFP阶段模块"""

    def __init__(self, in_channels):
        super().__init__()
        # 卷积组1: Conv3x3 -> BatchNorm -> ReLU
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # 卷积组2: Conv3x3 -> BatchNorm -> ReLU
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # 保留原始输入用于残差拼接

        # 卷积组1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 卷积组2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # 残差连接：将原始输入与当前输出拼接
        x = torch.cat([identity, x], dim=1)  # 沿通道维度拼接
        return x

class FFP(nn.Module):
    """完整的FFP模型"""

    def __init__(self, input_channels=3):
        super().__init__()
        # 四个处理阶段（Stage 1-4）
        self.stage1 = FFPStage(in_channels=input_channels)  # 输入3通道，输出3+64=67
        self.stage2 = FFPStage(in_channels=67)  # 输入67通道，输出67+64=131
        self.stage3 = FFPStage(in_channels=131)  # 输入131通道，输出131+64=195
        self.stage4 = FFPStage(in_channels=195)  # 输入195通道，输出195+64=259

        # 最终1x1卷积（输出4通道：HT/PT/BG/OV）
        self.final_conv = nn.Conv2d(259, 32, kernel_size=1)

    def forward(self, x):
        #print("FFP")
        # 顺序通过四个阶段
        x = self.stage1(x)  # 输出尺寸: (B, 67, H, W)
        x = self.stage2(x)  # 输出尺寸: (B, 131, H, W)
        x = self.stage3(x)  # 输出尺寸: (B, 195, H, W)
        x = self.stage4(x)  # 输出尺寸: (B, 259, H, W)

        # 最终卷积层
        x = self.final_conv(x)  # 输出尺寸: (B, 3, H, W)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffY < 0 or diffX < 0:
            # x1比x2大，裁剪x1
            x1 = x1[:, :, :x2.size(2), :x2.size(3)]
        else:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #print(x2.shape,x1.shape)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Refinement(nn.Module):
    def __init__(self, base):
        super().__init__()
        n_in_channel = 3
        cnum = 32
        ####downsapmle
        self.coarse_conva = ConvWithActivation(n_in_channel + base // 2, cnum, kernel_size=5, stride=1, padding=2)
        self.coarse_convb = ConvWithActivation(cnum, 2 * cnum, kernel_size=4, stride=2, padding=1)
        self.coarse_convc = ConvWithActivation(2 * cnum, 2 * cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_convd = ConvWithActivation(2 * cnum, 4 * cnum, kernel_size=4, stride=2, padding=1)
        self.coarse_conve = ConvWithActivation(4 * cnum, 4 * cnum, kernel_size=3, stride=1, padding=1)
        self.coarse_convf = ConvWithActivation(4 * cnum, 4 * cnum, kernel_size=3, stride=1, padding=1)
        ### astrous
        self.astrous_net = nn.Sequential(
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=2, padding=get_pad(cnum * 2, 3, 1, 2)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=4, padding=get_pad(cnum * 2, 3, 1, 4)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=8, padding=get_pad(cnum * 2, 3, 1, 8)),
            ConvWithActivation(4 * cnum, 4 * cnum, 3, 1, dilation=16, padding=get_pad(cnum * 2, 3, 1, 16)),
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
        self.c1 = nn.Conv2d(base // 2, cnum * 2, kernel_size=1)
        self.c2 = nn.Conv2d(base, cnum * 4, kernel_size=1)

    def forward(self,x, identity,con_x1,con_x2,shape0,shape1):
        #print("refinement")
        x = self.coarse_conva(torch.cat([x, identity], dim=1))  # shape0

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
        return x

class SinglePoolModule(nn.Module):
    def __init__(self, in_channels, kernel_size, pool=nn.AdaptiveAvgPool2d, normal=nn.BatchNorm2d):
        super(SinglePoolModule, self).__init__()
        out_channels = in_channels // 4
        self.conv = nn.Sequential(
            pool(kernel_size),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            normal(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        shape = x.shape
        x = self.conv(x)
        #print(x.shape)
        x = F.interpolate(x, size=shape[2:], mode='bilinear', align_corners=False)
        return x

class PPM(nn.Module):
    def __init__(self, in_channels, pool=nn.AdaptiveAvgPool2d, normal=nn.BatchNorm2d):
        super(PPM, self).__init__()

        self.pool1 = SinglePoolModule(in_channels, 1, pool, normal)
        self.pool2 = SinglePoolModule(in_channels, 2, pool, normal)
        self.pool3 = SinglePoolModule(in_channels, 4, pool, normal)
        self.pool4 = SinglePoolModule(in_channels, 8, pool, normal)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,padding=1),
            normal(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        #print("PPM")
        x1 = self.pool1(x)
        #print(x1.shape)
        x2 = self.pool2(x)
        #print(x2.shape)
        x3 = self.pool3(x)
        #print(x3.shape)
        x4 = self.pool4(x)
        #print(x4.shape)
        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.fuse_conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#-----------------------------------CA-----------------------------------#
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp,  reduction=8):
        super(CoordAtt, self).__init__()
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
#--------------------------------------------------------------------------#




#-----------------------------------CBAM-----------------------------------#
class CBAM(nn.Module):

    def __init__(self, n_channels_in, reduction_ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(n_channels_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, f):
        chan_att = self.channel_attention(f)
        # print(chan_att.size())
        fp = chan_att * f
        # print(fp.size())
        spat_att = self.spatial_attention(fp)
        # print(spat_att.size())
        fpp = spat_att * fp
        # print(fpp.size())
        return fpp

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, "Odd kernel size required"
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        # batchnorm

    def forward(self, x):
        max_pool = self.agg_channel(x, "max")
        avg_pool = self.agg_channel(x, "avg")
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1, x.size()[1], 1, 1)
        att = torch.sigmoid(conv)
        return att

    def agg_channel(self, x, pool="max"):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == "max":
            x = F.max_pool1d(x, c)
        elif pool == "avg":
            x = F.avg_pool1d(x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, n_channels_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.n_channels_in = n_channels_in
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.n_channels_in / float(self.reduction_ratio))

        self.bottleneck = nn.Sequential(
            nn.Linear(self.n_channels_in, self.middle_layer_size),
            nn.ReLU(),
            nn.Linear(self.middle_layer_size, self.n_channels_in)
        )

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel)
        max_pool = F.max_pool2d(x, kernel)

        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        max_pool = max_pool.view(max_pool.size()[0], -1)

        avg_pool_bck = self.bottleneck(avg_pool)
        max_pool_bck = self.bottleneck(max_pool)

        pool_sum = avg_pool_bck + max_pool_bck

        sig_pool = torch.sigmoid(pool_sum)
        sig_pool = sig_pool.unsqueeze(2).unsqueeze(3)

        out = sig_pool.repeat(1, 1, kernel[0], kernel[1])
        return out
#-------------------------------------------------------------------------#




#-----------------------------------ECA-----------------------------------#
class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_channels, k_size=3):
        out_channels = in_channels
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
#-------------------------------------------------------------------------#



#-----------------------------------ELA-----------------------------------#
class ELA(nn.Module):
    def __init__(self, in_channels, kernal_size=7):
        self.in_channels = in_channels
        self.out_channels = in_channels

        super().__init__()
        self.conv = nn.Conv1d(in_channels, self.out_channels, kernal_size, padding=kernal_size//2,groups = in_channels,bias = False)
        self.gn = nn.GroupNorm(8, self.out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b,c,h,w = x.size()

        x_h = torch.mean(x,dim = 3,keepdim=True).view(b,c,h)
        x_w = torch.mean(x,dim = 2,keepdim=True).view(b,c,w)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b,c,h,1)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b,c,1,w)
        return x*x_h*x_w
#-------------------------------------------------------------------------#




#-----------------------------------SA-----------------------------------#
class SA(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=4):
        super(SA, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1) #全局平均池化操作
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))#channel w
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))#channel b
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))#Spatial w
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))# Spatial b

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups)) # groupnorm

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        # print(x.shape)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
#-----------------------------------SA-----------------------------------#



#-----------------------------------SE-----------------------------------#
class SE(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SE, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def forward(self, X_input):
        b, c, _, _ = X_input.size()     # shape = [32, 64, 2000, 80]

        y = self.avg_pool(X_input)      # shape = [32, 64, 1, 1]
        y = y.view(b, c)                # shape = [32,64]

        # 第1个线性层（含激活函数），即公式中的W1，其维度是[channel, channer/16], 其中16是默认的
        y = self.linear1(y)             # shape = [32, 64] * [64, 4] = [32, 4]

        # 第2个线性层（含激活函数），即公式中的W2，其维度是[channel/16, channer], 其中16是默认的
        y = self.linear2(y)             # shape = [32, 4] * [4, 64] = [32, 64]
        y = y.view(b, c, 1, 1)          # shape = [32, 64, 1, 1]， 这个就表示上面公式的s, 即每个通道的权重

        return X_input*y.expand_as(X_input)
#-------------------------------------------------------------------------#
