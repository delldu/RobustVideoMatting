"""Create model."""
# coding=utf-8


import torch
from torch import nn
from torch.nn import functional as F

from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from typing import List

import pdb


class MattingNetwork(nn.Module):
    def __init__(self, backbone: str = "mobilenetv3", pretrained_backbone: bool = False):
        super().__init__()
        assert backbone in ["mobilenetv3", "resnet50"]

        if backbone == "mobilenetv3":
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])

        self.project_mat = Projection(16, 4)  # NOT USED !!!
        self.project_seg = Projection(16, 1)

        self.refiner = DeepGuidedFilterRefiner()
        # pretrained_backbone = False

        self.r1 = torch.zeros(1, 1, 1, 1)
        self.r2 = torch.zeros(1, 1, 1, 1)
        self.r3 = torch.zeros(1, 1, 1, 1)
        self.r4 = torch.zeros(1, 1, 1, 1)

    def forward(self, src):
        # src.size() -- [1, 3, 1080, 1920]
        B, C, H, W = src.shape

        f1, f2, f3, f4 = self.backbone(src)
        # f1.size(), f2.size(), f3.size(), f4.size()
        # [1, 64, 144, 256], [1, 256, 72, 128], [1, 512, 36, 64], [1, 2048, 18, 32]

        f4 = self.aspp(f4)
        hid, self.r1, self.r2, self.r3, self.r4 = self.decoder(src, f1, f2, f3, f4, self.r1, self.r2, self.r3, self.r4)
        # hid.size() -- [1, 16, 288, 512]
        # type(rec), len(rec), rec[0].size(), rec[1].size(), rec[2].size(), rec[3].size()
        # (<class 'list'>, 4,
        # [1, 16, 144, 256], [1, 32, 72, 128], [1, 64, 36, 64], [1, 128, 18, 32]

        # seg = self.project_seg(hid)  # .clamp(0, 1.0)
        # mask = F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)
        # # bg = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1).to(src.device)
        # # output = mask * src + (1.0 - mask) * bg
        # output = torch.cat((src, mask), dim=1)
        # return output

        src_residual, mask = self.project_mat(hid).split([3, 1], dim=-3)
        mask = mask.clamp(0.0, 1.0)
        output = torch.cat((src, mask), dim=1)
        return output

"""
Adopted from <https://github.com/wuhuikai/DeepGuidedFilter/>
"""
class DeepGuidedFilterRefiner(nn.Module):
    def __init__(self, hid_channels=16):
        super().__init__()
        self.box_filter = nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False, groups=4)
        self.box_filter.weight.data[...] = 1 / 9
        self.conv = nn.Sequential(
            nn.Conv2d(4 * 2 + hid_channels, hid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, hid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(True),
            nn.Conv2d(hid_channels, 4, kernel_size=1, bias=True),
        )

    def forward(self, fine_src, base_src, base_fgr, base_pha, base_hid) -> List[torch.Tensor]:
        # fine_src.size(), base_src.size(), base_fgr.size(), base_pha.size(), base_hid.size()
        # [1, 3, 1080, 1920], [1, 3, 288, 512], [1, 3, 288, 512],[1, 1, 288, 512],
        # [1, 16, 288, 512]

        fine_x = torch.cat([fine_src, fine_src.mean(1, keepdim=True)], dim=1)
        base_x = torch.cat([base_src, base_src.mean(1, keepdim=True)], dim=1)
        base_y = torch.cat([base_fgr, base_pha], dim=1)

        mean_x = self.box_filter(base_x)
        mean_y = self.box_filter(base_y)
        cov_xy = self.box_filter(base_x * base_y) - mean_x * mean_y
        var_x = self.box_filter(base_x * base_x) - mean_x * mean_x

        A = self.conv(torch.cat([cov_xy, var_x, base_hid], dim=1))
        b = mean_y - A * mean_x

        H, W = fine_src.shape[2:]
        A = F.interpolate(A, (H, W), mode="bilinear", align_corners=False)
        b = F.interpolate(b, (H, W), mode="bilinear", align_corners=False)

        out = A * fine_x + b
        fgr, pha = out.split([3, 1], dim=1)
        return fgr, pha


class LRASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )
        self.aspp2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.Sigmoid()
        )
        # in_channels = 2048
        # out_channels = 256

    def forward(self, x):
        return self.aspp1(x) * self.aspp2(x)


class RecurrentDecoder(nn.Module):
    def __init__(self, feature_channels, decoder_channels):
        super().__init__()
        self.avgpool = AvgPool()
        self.decode4 = BottleneckBlock(feature_channels[3])
        self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])
        # feature_channels = [64, 256, 512, 256]
        # decoder_channels = [128, 64, 32, 16]

    def forward(self, s0, f1, f2, f3, f4, r1, r2, r3, r4) -> List[torch.Tensor]:
        # pp s0.size(), f1.size(), f2.size(), f3.size(), f4.size()
        # ([1, 3, 288, 512],
        # [1, 64, 144, 256],
        # [1, 256, 72, 128],
        # [1, 512, 36, 64],
        # [1, 256, 18, 32])
        s1, s2, s3 = self.avgpool(s0)
        x4, r4 = self.decode4(f4, r4)
        x3, r3 = self.decode3(x4, f3, s3, r3)
        x2, r2 = self.decode2(x3, f2, s2, r2)
        x1, r1 = self.decode1(x2, f1, s1, r1)
        x0 = self.decode0(x1, s0)
        return x0, r1, r2, r3, r4


class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AvgPool2d(2, 2, count_include_pad=False, ceil_mode=True)

    def forward(self, s0) -> List[torch.Tensor]:
        s1 = self.avgpool(s0)
        s2 = self.avgpool(s1)
        s3 = self.avgpool(s2)
        return s1, s2, s3


class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.gru = ConvGRU(channels // 2)

    def forward(self, x, r) -> List[torch.Tensor]:
        a, b = x.split(self.channels // 2, dim=-3)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=-3)
        return x, r


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, src_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.gru = ConvGRU(out_channels // 2)

    def forward(self, x, f, s, r) -> List[torch.Tensor]:
        x = self.upsample(x)
        x = x[:, :, : s.size(2), : s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)
        a, b = x.split(self.out_channels // 2, dim=1)
        b, r = self.gru(b, r)
        x = torch.cat([a, b], dim=1)
        return x, r


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x, s):
        x = self.upsample(x)
        x = x[:, :, : s.size(2), : s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        return x


class ConvGRU(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding), nn.Sigmoid())
        self.hh = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size, padding=padding), nn.Tanh())

    def forward(self, x, h) -> List[torch.Tensor]:
        if h.size() != x.size():
            # torch.Size([1,1,1,1]):
            h = torch.zeros_like(x)

        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)
