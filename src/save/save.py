from monai.transforms import (
    SaveImage
    
)
import torch
import torch.nn as nn
import os


class WriteNifti(nn.Module):
    def __init__(
        self,
        defaults_path,
        model_name,
        loss_function_name,
        overlap,
        inference_mode,
        to_onehot,
        val_inputs,
        val_labels_,
        val_outputs_,

    ) -> None:
        super(WriteNifti, self).__init__()
        # super().__init__()
        self.defaults_path = defaults_path
        self.model_name = model_name
        self.loss_function_name = loss_function_name
        self.overlap = overlap
        self.inference_mode = inference_mode
        self.to_onehot = to_onehot
        self.val_inputs = val_inputs
        self.val_labels_ = val_labels_
        self.val_outputs_ = val_outputs_

    def save(self):
        save_path = os.path.join(self.defaults_path + '/results/' + self.model_name + '_' + self.loss_function_name + '_' + str(self.overlap) + '_' + self.inference_mode)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if (self.to_onehot == 1) or (self.to_onehot == None):
            val_inputs = self.val_inputs[0][0]
            val_labels_ = self.val_labels_[0][0]
            val_outputs_ = self.val_outputs_[0][1]

            val_outputs_[val_outputs_>= 0.5] = 1.
            val_outputs_[val_outputs_ < 0.5] = 0.  
   

            SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="ori",print_log=True,padding_mode="zeros")(val_inputs)
            SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="label",print_log=True,padding_mode="zeros")(val_labels_)
            SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="pred",print_log=True,padding_mode="zeros")(val_outputs_)
        else:
            for i in len(val_labels_.shape[1]):
                val_outputs_ = val_outputs_[:,i,:,:,:]
                val_labels_ = val_labels_[:,i,:,:,:]
                val_inputs = val_inputs[0][0]
                val_outputs_[val_outputs_>= 0.5] = 1.
                val_outputs_[val_outputs_ < 0.5] = 0. 
                SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="ori",print_log=True,padding_mode="zeros")(val_inputs)
                SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="_label",print_log=True,padding_mode="zeros")(val_labels_)
                SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="_pred",print_log=True,padding_mode="zeros")(val_outputs_)

    def forward(self):
        self.save()
