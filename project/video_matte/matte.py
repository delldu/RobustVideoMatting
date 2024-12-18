"""Create model."""
# coding=utf-8

import os
import torch
from torch import nn
from torch.nn import functional as F

from .mobilenetv3 import MobileNetV3LargeEncoder
# from .resnet import ResNet50Encoder
from typing import List
import todos

import pdb

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
        # 960, 128
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
        # feature_channels = [16, 24, 40, 128]
        # decoder_channels = [80, 40, 32, 16]
        self.avgpool = AvgPool()
        # self.decode4 = BottleneckBlock(feature_channels[3])
        # self.decode3 = UpsamplingBlock(feature_channels[3], feature_channels[2], 3, decoder_channels[0])
        # self.decode2 = UpsamplingBlock(decoder_channels[0], feature_channels[1], 3, decoder_channels[1])
        # self.decode1 = UpsamplingBlock(decoder_channels[1], feature_channels[0], 3, decoder_channels[2])
        # self.decode0 = OutputBlock(decoder_channels[2], 3, decoder_channels[3])

        self.decode4 = BottleneckBlock(128)
        self.decode3 = UpsamplingBlock(128, 40, 3, 80)
        self.decode2 = UpsamplingBlock(80, 24, 3, 40)
        self.decode1 = UpsamplingBlock(40, 16, 3, 32)
        self.decode0 = OutputBlock(32, 3, 16)



    def forward(self, s0, f1, f2, f3, f4):

        s1, s2, s3 = self.avgpool(s0)
        x4 = self.decode4(f4)
        x3 = self.decode3(x4, f3, s3)
        x2 = self.decode2(x3, f2, s2)
        x1 = self.decode1(x2, f1, s1)
        x0 = self.decode0(x1, s0)
        return x0 #, r1, r2, r3, r4


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

    def forward(self, x) -> List[torch.Tensor]:
        a, b = x.split(self.channels // 2, dim=-3)
        b = self.gru(b)
        x = torch.cat([a, b], dim=-3)
        return x


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

    def forward(self, x, f, s) -> List[torch.Tensor]:
        x = self.upsample(x)
        x = x[:, :, : s.size(2), : s.size(3)]
        x = torch.cat([x, f, s], dim=1)
        x = self.conv(x)

        a, b = x.split(self.out_channels // 2, dim=1)
        b = self.gru(b)
        x = torch.cat([a, b], dim=1)

        return x


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
    def __init__(self, channels):
        super().__init__()
        kernel_size = 3
        self.channels = channels
        self.ih = nn.Sequential(nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=1), nn.Sigmoid())
        self.hh = nn.Sequential(nn.Conv2d(channels * 2, channels, kernel_size, padding=1), nn.Tanh())

    def forward(self, x):
        # x.size: [1, 64, 68, 120], h.size: [1, 1, 1, 1]
        h = torch.zeros_like(x)
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        # tensor [r] size: [1, 64, 68, 120], min: 0.0, max: 1.0, mean: 0.358712
        # tensor [z] size: [1, 64, 68, 120], min: 0.0, max: 1.0, mean: 0.405152
        c = self.hh(torch.cat([x, h], dim=1))
        h = z * c
        return h


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 16, 4
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class MattingNetwork(nn.Module):
    # def __init__(self, backbone: str = "mobilenetv3", pretrained_backbone: bool = False):
    def __init__(self):
        super().__init__()
        # Define max GPU/CPU memory -- 4G
        self.MAX_H = 2048
        self.MAX_W = 4048
        self.MAX_TIMES = 4
        # GPU 8G, 220ms

        self.backbone = MobileNetV3LargeEncoder()
        self.aspp = LRASPP(960, 128)
        self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])

        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)  # NOT USED !!!

        self.refiner = DeepGuidedFilterRefiner() # NOT USED !!!
        # pretrained_backbone = False
        self.load_weights()

        del self.refiner
        del self.project_seg
        # from ggml_engine import create_network
        # create_network(self)
        # torch.save(self.state_dict(), "/tmp/a.pth")


    def load_weights(self, model_path="models/video_matte.pth"):
        print(f"Loading weights from {model_path} ......")
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        sd = torch.load(checkpoint)
        self.load_state_dict(sd)        


    def forward(self, src):
        f1, f2, f3, f4 = self.backbone(src)
        f4 = self.aspp(f4)
        hid = self.decoder(src, f1, f2, f3, f4)
        src_residual, mask = self.project_mat(hid).split([3, 1], dim=-3)
        mask = mask.clamp(0.0, 1.0)
        output = torch.cat((src, mask), dim=1)
        return output
