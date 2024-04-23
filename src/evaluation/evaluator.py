from monai.metrics import DiceMetric, HausdorffDistanceMetric

import torch.nn as nn
import time

class Evaluator(nn.Module):
    def __init__(
        self,
        start_time,
        val_output_convert,
        val_labels_convert,
        step

    ) -> None:
        super(Evaluator, self).__init__()
        # super().__init__()
        self.start_time = start_time
        self.val_output_convert = val_output_convert
        self.val_labels_convert = val_labels_convert
        self.step = step

    def evaluation(self):
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
        hd_metric = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False, percentile = 95.0)

        prediction_time = time.time() - self.start_time
        dice = dice_metric(y_pred=self.val_output_convert, y=self.val_labels_convert)
        hd95 = hd_metric(y_pred=self.val_output_convert, y=self.val_labels_convert)
                    
        print('Case No. ::' , str(self.step+1), ',', 'Prediction time ::', prediction_time)
        print('Dice_metric ::', dice)
        print('HD_metric ::', hd95)

    def forward(self):
        self.evaluation()
