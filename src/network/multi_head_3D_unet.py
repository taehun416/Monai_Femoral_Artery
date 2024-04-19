# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.layers.factories import Act, Norm
from monai.utils import alias, export
from typing import Tuple

from src.network import UNet


__all__ = ["MultiHeadUNet", "MultiheaduNet"]

@export("monai.networks.nets")
@alias("Unet")

class MultiHeadUNet(nn.Module):
    def __init__(
        self, 
        spatial_dims: int, 
        in_channels: int, 
        out_channels: int, 
        channels: Sequence[int], 
        strides: Sequence[int], 
        kernel_size: Sequence[int] | int = 3, 
        up_kernel_size: Sequence[int] | int = 3, 
        num_res_units: int = 0, 
        act: tuple | str = Act.PRELU, 
        norm: tuple | str = Norm.INSTANCE, 
        dropout: float = 0.0, 
        bias: bool = True, 
        adn_ordering: str = "NDA") -> None:

        super().__init__()
        self.unet = UNet(spatial_dims, in_channels, out_channels, channels, strides, kernel_size, up_kernel_size, num_res_units, act, norm, dropout, bias, adn_ordering)
        # Additional layers for multi-head output
        self.head1 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.head2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.unet(x)
        outputs = self.head1(x)
        outputs_dist = self.head2(x)
        return outputs, outputs_dist
    
MultiheaduNet = MultiHeadUNet

