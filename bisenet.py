import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import Resnet18


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = ks // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch, ks=3)
        self.conv_atten = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True)
        self.bn_atten = nn.BatchNorm2d(out_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        atten = F.adaptive_avg_pool2d(x, 1)
        atten = self.bn_atten(self.conv_atten(atten))
        atten = self.sigmoid(atten)
        return x * atten


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet18()  # returns feat8(128ch), feat16(256ch), feat32(512ch)

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)

        self.conv_head16 = ConvBNReLU(128, 128, ks=3)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3)

        # conv_avg in checkpoint is 1x1, from 512 -> 128
        self.conv_avg = ConvBNReLU(512, 128, ks=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)          # 1/8(128), 1/16(256), 1/32(512)

        feat32_arm = self.arm32(feat32)                 # -> 128
        avg = F.adaptive_avg_pool2d(feat32, 1)          # on 512-ch feature
        avg = self.conv_avg(avg)                        # 512->128, 1x1
        feat32_sum = feat32_arm + avg                   # 128

        up32 = F.interpolate(feat32_sum, size=feat16.shape[2:], mode='bilinear', align_corners=True)

        feat16_arm = self.arm16(feat16)                 # 256->128
        feat16_sum = feat16_arm + up32                  # 128
        feat16_head = self.conv_head16(feat16_sum)      # 128
        feat32_head = self.conv_head32(feat32_arm)      # 128

        up16 = F.interpolate(feat16_head, size=feat8.shape[2:], mode='bilinear', align_corners=True)

        return feat8, up16, feat16_head, feat32_head    # (128ch @1/8), (128ch @1/8), (128ch @1/16), (128ch @1/32)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # checkpoint uses 1x1 conv here: weight [out_ch, in_ch, 1, 1]
        self.convblk = ConvBNReLU(in_ch, out_ch, ks=1, padding=0)
        self.conv1 = nn.Conv2d(out_ch, out_ch // 4, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch // 4, out_ch, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        feat = self.convblk(torch.cat([x, y], dim=1))   # (B, out_ch, H, W)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.relu(self.conv1(atten))
        atten = self.sigmoid(self.conv2(atten))
        return feat * atten + feat


class BiSeNetOutput(nn.Module):
    def __init__(self, in_ch, mid_ch, n_classes):
        super().__init__()
        # checkpoint expects conv_out.*.conv.* for a 3x3 conv, then conv_out.*.conv_out as 1x1
        self.conv = ConvBNReLU(in_ch, mid_ch, ks=3)
        self.conv_out = nn.Conv2d(mid_ch, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = ContextPath()
        # feat8 (128ch) + up16 (128ch) -> 256
        self.ffm = FeatureFusionModule(256, 256)

        # heads: main + two auxiliary
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 256, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 256, n_classes)

    def forward(self, x):
        feat8, up16, feat16_head, feat32_head = self.cp(x)
        fusion = self.ffm(feat8, up16)

        out = self.conv_out(fusion)
        out16 = self.conv_out16(feat16_head)
        out32 = self.conv_out32(feat32_head)

        return [out, out16, out32]