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

from .cldice import SoftclDiceLoss, SoftDiceclDiceLoss
from .dice import (
    Dice,
    DiceCELoss,
    DiceFocalLoss,
    DiceLoss,
    GeneralizedDiceFocalLoss,
    GeneralizedDiceLoss,
    GeneralizedWassersteinDiceLoss,
    MaskedDiceLoss,
    dice_ce,
    dice_focal,
    generalized_dice,
    generalized_dice_focal,
    generalized_wasserstein_dice,
)

from .fgdtmloss import FGDTMloss