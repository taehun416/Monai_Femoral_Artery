from monai.transforms import (
    AsDiscrete
    
)
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import torch
import torch.nn as nn
import os

import time

from src.save.save import WriteNifti
from src.evaluation.evaluator import Evaluator

class Inference(nn.Module):
    def __init__(
        self,
        model,
        model_name,
        loss_function_name,
        defaults_path,
        val_loader,
        overlap,
        inference_mode,
        to_onehot,
        device,
        save_flag,
        evaluation_flag,
        amp: bool = True
        

    ) -> None:
        super(Inference, self).__init__()
        # super().__init__()
        self.model = model
        self.model_name = model_name
        self.loss_function_name = loss_function_name
        self.defaults_path = defaults_path
        self.val_loader = val_loader
        self.overlap = overlap
        self.inference_mode = inference_mode
        self.to_onehot = to_onehot
        self.device = device
        self.amp = amp
        self.save_flag = save_flag
        self.evaluation_flag = evaluation_flag

    def inference(self):
        post_label = AsDiscrete(to_onehot=self.to_onehot)
        post_pred = AsDiscrete(argmax=True, to_onehot=self.to_onehot)
        device_cpu = torch.device('cpu')
        self.model.load_state_dict(torch.load(os.path.join(self.defaults_path + 'weight/' + self.model_name + '_' + self.loss_function_name + '/best_metric_model.pth')))
        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                start_time = time.time()
                val_inputs, val_labels = (batch["image"].to(self.device), batch["label"].to(self.device))
                if self.amp:
                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model, self.overlap, mode = self.inference_mode)
                else:
                    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model, self.overlap, mode = self.inference_mode)
                    
                val_outputs_ = val_outputs.to(device_cpu)
                val_labels_ = val_labels.to(device_cpu)
                    
                val_labels_list = decollate_batch(val_labels_)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs_)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]

                if self.evaluation_flag ==1:
                    Evaluator(start_time, val_output_convert, val_labels_convert, step).forward()

                if self.save_flag ==1:
                    WriteNifti(self.defaults_path, self.model_name, self.loss_function_name, self.overlap, self.inference_mode, self.to_onehot, val_inputs, val_labels_, val_outputs_).forward()
    
    def forward(self):
        self.inference()

class InferenceMode(nn.Module):
    def __init__(
        self,
        model,
        model_name,
        loss_function_name,
        defaults_path,
        val_loader,
        overlap,
        to_onehot,
        device,
        save_flag,
        evaluation_flag,
        amp: bool = True
        

    ) -> None:
        super(Inference, self).__init__()
        # super().__init__()
        self.model = model
        self.model_name = model_name
        self.loss_function_name = loss_function_name
        self.defaults_path = defaults_path
        self.val_loader = val_loader
        self.overlap = overlap
        self.to_onehot = to_onehot
        self.device = device
        self.amp = amp
        self.save_flag = save_flag
        self.evaluation_flag = evaluation_flag

    def inference(self):
        post_label = AsDiscrete(to_onehot=self.to_onehot)
        post_pred = AsDiscrete(argmax=True, to_onehot=self.to_onehot)
        device_cpu = torch.device('cpu')
        self.model.load_state_dict(torch.load(os.path.join(self.defaults_path + 'weight/' + self.model_name + '_' + self.loss_function_name + '/best_metric_model.pth')))
        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                start_time = time.time()
                val_inputs, val_labels = (batch["image"].to(self.device), batch["label"].to(self.device))
                if self.amp:
                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model, self.overlap)
                else:
                    val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model, self.overlap)
                    
                val_outputs_ = val_outputs.to(device_cpu)
                val_labels_ = val_labels.to(device_cpu)
                    
                val_labels_list = decollate_batch(val_labels_)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs_)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]

                if self.evaluation_flag ==1:
                    Evaluator(start_time, val_output_convert, val_labels_convert, step).forward()

                if self.save_flag ==1:
                    WriteNifti(self.defaults_path, self.model_name, self.loss_function_name, self.overlap, self.to_onehot, val_inputs, val_labels_, val_outputs_).forward()
    
    def forward(self):
        self.inference()


class Inference_aug_test(nn.Module):
    def __init__(
        self,
        model,
        model_name,
        loss_function_name,
        defaults_path,
        val_loader,
        overlap,
        inference_mode,
        to_onehot,
        device,
        crop_size_x,
        crop_size_y,
        crop_size_z,
        save_flag,
        evaluation_flag,
        amp: bool = True
        

    ) -> None:
        super(Inference_aug_test, self).__init__()
        # super().__init__()
        self.model = model
        self.model_name = model_name
        self.loss_function_name = loss_function_name
        self.defaults_path = defaults_path
        self.val_loader = val_loader
        self.overlap = overlap
        self.inference_mode = inference_mode
        self.to_onehot = to_onehot
        self.device = device
        self.crop_size_x = crop_size_x
        self.crop_size_y = crop_size_y
        self.crop_size_z = crop_size_z
        self.amp = amp
        self.save_flag = save_flag
        self.evaluation_flag = evaluation_flag

    def inference(self):
        post_label = AsDiscrete(to_onehot=self.to_onehot)
        post_pred = AsDiscrete(argmax=True, to_onehot=self.to_onehot)
        device_cpu = torch.device('cpu')
        self.model.load_state_dict(torch.load(os.path.join(self.defaults_path + 'weight/' + self.model_name + '_' + self.loss_function_name + '/best_metric_model.pth')))
        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                start_time = time.time()
                val_inputs, val_labels = (batch["image"].to(self.device), batch["label"].to(self.device))
                if self.amp:
                    with torch.cuda.amp.autocast():
                        val_outputs = sliding_window_inference(val_inputs, (self.crop_size_x, self.crop_size_y, self.crop_size_z), 4, self.model, self.overlap, mode = self.inference_mode)
                else:
                    val_outputs = sliding_window_inference(val_inputs, (self.crop_size_x, self.crop_size_y, self.crop_size_z), 4, self.model, self.overlap, mode = self.inference_mode)
                    
                val_outputs_ = val_outputs.to(device_cpu)
                val_labels_ = val_labels.to(device_cpu)
                    
                val_labels_list = decollate_batch(val_labels_)
                val_labels_convert = [
                    post_label(val_label_tensor) for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs_)
                val_output_convert = [
                    post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
                ]

                if self.evaluation_flag ==1:
                    Evaluator(start_time, val_output_convert, val_labels_convert, step).forward()

                if self.save_flag ==1:
                    WriteNifti(self.defaults_path, self.model_name, self.loss_function_name, self.overlap, self.inference_mode, self.to_onehot, val_inputs, val_labels_, val_outputs_).forward()
    
    def forward(self):
        self.inference()