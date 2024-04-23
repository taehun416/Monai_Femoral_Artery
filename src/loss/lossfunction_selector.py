from src.loss import DiceLoss, DiceCELoss, DiceFocalLoss, SoftclDiceLoss, FGDTMloss

class LossFunctionSelector:
    def __init__(self, loss_name):
        self.loss_name = loss_name

    def get_loss_function(self):
        if self.loss_name == 'DiceLoss':
            return self.get_dice_loss()
        elif self.loss_name == 'DiceCELoss':
            return self.get_dice_ce_loss()
        elif self.loss_name == 'FGDTMloss':
            return self.get_fgdtm_loss()
        elif self.loss_name == 'DiceFocalLoss':
            return self.get_dice_focal_loss()
        elif self.loss_name == 'SoftclDiceLoss':
            return self.get_softcl_dice_loss()
        else:
            raise ValueError(f"Unsupported loss function name: {self.loss_name}")

    def get_dice_loss(self):
        return DiceLoss(to_onehot_y=True, softmax=True)

    def get_dice_ce_loss(self):
        return DiceCELoss(to_onehot_y=True, softmax=True)

    def get_fgdtm_loss(self):
        return FGDTMloss()

    def get_dice_focal_loss(self):
        return DiceFocalLoss(to_onehot_y=True, softmax=True)

    def get_softcl_dice_loss(self):
        return SoftclDiceLoss()
    
    