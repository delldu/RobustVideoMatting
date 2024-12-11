import torch
from torch import nn
# from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidualConfig
from .local_v3 import MobileNetV3, InvertedResidualConfig

from torchvision.transforms.functional import normalize
from typing import List

import pdb
# https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py

class MobileNetV3LargeEncoder(MobileNetV3):
    def __init__(self):
        super().__init__(
            inverted_residual_setting=[
                # input_channels, kernel, expanded_channels, out_channels,
                # use_se: bool, # activation: str,
                # stride, dilation, width_mult
                InvertedResidualConfig(16, 3, 16, 16, False, "RE", 1, 1, 1),
                InvertedResidualConfig(16, 3, 64, 24, False, "RE", 2, 1, 1),  # C1
                InvertedResidualConfig(24, 3, 72, 24, False, "RE", 1, 1, 1),
                InvertedResidualConfig(24, 5, 72, 40, True, "RE", 2, 1, 1),  # C2
                InvertedResidualConfig(40, 5, 120, 40, True, "RE", 1, 1, 1),
                InvertedResidualConfig(40, 5, 120, 40, True, "RE", 1, 1, 1),
                InvertedResidualConfig(40, 3, 240, 80, False, "HS", 2, 1, 1),  # C3
                InvertedResidualConfig(80, 3, 200, 80, False, "HS", 1, 1, 1),
                InvertedResidualConfig(80, 3, 184, 80, False, "HS", 1, 1, 1),
                InvertedResidualConfig(80, 3, 184, 80, False, "HS", 1, 1, 1),
                InvertedResidualConfig(80, 3, 480, 112, True, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 3, 672, 112, True, "HS", 1, 1, 1),
                InvertedResidualConfig(112, 5, 672, 160, True, "HS", 2, 2, 1),  # C4
                InvertedResidualConfig(160, 5, 960, 160, True, "HS", 1, 2, 1),
                InvertedResidualConfig(160, 5, 960, 160, True, "HS", 1, 2, 1),
            ],
            last_channel=1280,
        )
        del self.avgpool
        del self.classifier


    def forward(self, x) -> List[torch.Tensor]:
        x = normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        x = self.features[0](x)
        x = self.features[1](x)
        f1 = x
        x = self.features[2](x)
        x = self.features[3](x)
        f2 = x
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        f3 = x
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        f4 = x

        return [f1, f2, f3, f4]

