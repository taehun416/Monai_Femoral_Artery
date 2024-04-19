from scipy.ndimage import distance_transform_edt as distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from monai.networks import one_hot

class FGDTMloss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FGDTMloss, self).__init__()

    def compute_dtm(self, img_gt, out_shape):
        """
        compute the distance transform map of foreground in binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the foreground Distance Map (SDM) 
        dtm(x) = 0; x in segmentation boundary
                inf|x-y|; x in segmentation
        """
        fg_dtm = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
            for c in range(out_shape[1]):
                posmask = img_gt[b].astype(bool)
                if posmask.any():
                    posdis = distance(posmask)
                    fg_dtm[b][c] = posdis

        # for b in range(out_shape[0]): # batch size
        #     posmask = img_gt[b].astype(bool)
        #     if posmask.any():
        #         posdis = distance(posmask)
        #         fg_dtm[b] = posdis

        return fg_dtm

    def dice_loss(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        N = inputs.size()[0]
        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(N, -1)
        targets = targets.contiguous().view(N, -1)

        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)

        return 1 - dice.sum() / N

    def weighted_l1_loss(self, input, target, mask):
        # Compute the absolute differences
        abs_diff = torch.abs(input - target)
        # Apply the binary mask (element-wise multiplication)
        weighted_diff = abs_diff * mask
        # Check if the mask is zero everywhere
        if torch.sum(mask) == 0:
            return torch.tensor(0.0)  # Return zero loss if the mask has no valid elements
        # Compute the average of the weighted differences
        loss = torch.sum(weighted_diff) / torch.sum(mask)  # Normalize by the number of non-zero weights
        return loss

    def forward(self, outputs, outputs_dist, y, distance_map_weight, smooth=1):
    # def forward(self, outputs: torch.Tensor, outputs_dist: torch.Tensor, y, distance_map_weight, smooth=1) -> torch.Tensor:

        # comment out if your model contains a sigmoid or equivalent activation layer
        assert isinstance(distance_map_weight, float), "distance_map_weight is not a float"

        n_pred_ch = outputs.shape[1]
        y = one_hot(y, num_classes=n_pred_ch)

        outputs = outputs[:,1:,:,:,:]
        y = y[:,1:,:,:,:]
        outputs_dist = outputs_dist[:,1:,:,:,:]

        gt_dis = self.compute_dtm(y.cpu().numpy(), outputs_dist.shape)
        gt_dis = torch.from_numpy(gt_dis).float().cuda()
        
        loss_ce = F.binary_cross_entropy_with_logits(outputs, y)
        loss_dice = self.dice_loss(outputs, y)

        loss_dist = self.weighted_l1_loss(outputs_dist, gt_dis, y) 


        loss = loss_ce + loss_dice + distance_map_weight * loss_dist


        return loss
    