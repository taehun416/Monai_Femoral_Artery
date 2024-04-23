from monai.transforms import (
    SaveImage
    
)
import torch
import torch.nn as nn
import os
from scipy.ndimage import distance_transform_edt as distance
import numpy as np
import nibabel as nib


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

class ValidWriteNifti(nn.Module):
    def __init__(
        self,
        defaults_path,
        model_name,
        loss_function_name,
        to_onehot,
        train_inputs,
        train_labels_,
        train_outputs_,
        step

    ) -> None:
        super(ValidWriteNifti, self).__init__()
        # super().__init__()
        self.defaults_path = defaults_path
        self.model_name = model_name
        self.loss_function_name = loss_function_name
        self.to_onehot = to_onehot
        self.train_inputs = train_inputs
        self.train_labels_ = train_labels_
        self.train_outputs_ = train_outputs_
        self.step = step

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

        return fg_dtm
    
    def save(self):
        save_path = os.path.join(self.defaults_path + '/results_valid/' + str(self.step) + '_' + self.model_name + '_' + self.loss_function_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if (self.to_onehot == 1) or (self.to_onehot == None):

            train_outputs_ = self.train_outputs_

            label_dist = self.compute_dtm(self.train_labels_.cpu().numpy(), self.train_outputs_.shape)
            label_dist = torch.from_numpy(label_dist).float().to(self.train_outputs_.device)
            train_outputs_[train_outputs_>= 0.5] = 1.
            train_outputs_[train_outputs_ < 0.5] = 0.  

            train_inputs = nib.Nifti1Image(self.train_inputs.cpu().detach().numpy()[0][0], np.eye(4))
            nib.save(train_inputs, os.path.join(save_path + '/img'))
            train_labels_ = nib.Nifti1Image(self.train_labels_.cpu().detach().numpy()[0][0], np.eye(4))
            nib.save(train_labels_, os.path.join(save_path + '/label'))
            train_outputs_ = nib.Nifti1Image(train_outputs_.cpu().detach().numpy()[0][0], np.eye(4))
            nib.save(train_outputs_, os.path.join(save_path + '/pred'))
            label_dist = nib.Nifti1Image(label_dist.cpu().detach().numpy()[0][0], np.eye(4))
            nib.save(label_dist, os.path.join(save_path + '/label_dist' ))

        else:
            for i in len(train_labels_.shape[1]):
                train_inputs = train_inputs[:,i,:,:,:]
                train_labels_ = train_labels_[:,i,:,:,:]
                train_outputs_ = train_outputs_[0][0]
                # val_outputs_[val_outputs_>= 0.5] = 1.
                # val_outputs_[val_outputs_ < 0.5] = 0. 
                SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="ori",print_log=True,padding_mode="zeros")(train_inputs)
                SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="_label",print_log=True,padding_mode="zeros")(train_labels_)
                SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="_pred",print_log=True,padding_mode="zeros")(train_outputs_)
                SaveImage(output_dir=save_path, output_ext=".nii.gz", output_postfix="dist",print_log=True,padding_mode="zeros")(label_dist)

    def forward(self):
        self.save()