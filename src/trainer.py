from monai.transforms import (
    AsDiscrete,
    Compose,
)
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from src.metric.metric import Cal_metrics
import numpy as np

class Trainer:
    def __init__(self, model_name, model, loss_function_name, loss_function, dist_flag, distance_map_weight, max_iterations, eval_num, optimizer_name, defaults_path, train_loader, val_loader, to_onehot):
        self.model_name = model_name
        self.model = model
        self.loss_function_name = loss_function_name
        self.loss_function = loss_function
        self.dist_flag = dist_flag
        self.max_iterations = max_iterations
        self.eval_num = eval_num
        self.optimizer_name = optimizer_name
        self.defaults_path = defaults_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.to_onehot = to_onehot
        self.distance_map_weight = distance_map_weight

        self.best_metric = -1

    def optimizer_list(self):
        if self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
        if self.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

        return optimizer

    def validation(self, global_step, epoch_iterator_val):
        post_label = AsDiscrete(to_onehot=self.to_onehot)
        post_pred = AsDiscrete(argmax=True, to_onehot=self.to_onehot)
        dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

        self.model.eval()
        with torch.no_grad():
            for batch in epoch_iterator_val:
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model)
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                dice_metric(y_pred=val_output_convert, y=val_labels_convert)
                epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
            mean_dice_val = dice_metric.aggregate().item()
            dice_metric.reset()
        return mean_dice_val
        
    def train(self,global_step, dice_val_best, global_step_best):
        epoch_loss = 0
        step = 0
        epoch_loss_values = []
        metric_values = []
        epoch_iterator = tqdm(self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            self.model.train()
            step += 1
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            if self.dist_flag == 0:
                logit_map = self.model(x)
                loss = self.loss_function(logit_map, y)
            else:
                logit_map, outputs_dist = self.model(x)
                outputs_dist = nn.Softplus()(outputs_dist)
                loss = self.loss_function(logit_map, outputs_dist, y, self.distance_map_weight)

            epoch_loss += loss.item()
            optimizer = self.optimizer_list()
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_description(  # noqa: B038
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, self.max_iterations, loss)
            )
            if (global_step % self.eval_num == 0 and global_step != 0) or global_step == self.max_iterations:
                epoch_iterator_val = tqdm(self.val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                dice_val = self.validation(global_step, epoch_iterator_val)
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                metric_values.append(dice_val)
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    if not os.path.exists(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/')):
                        os.makedirs(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' ))
                    torch.save(self.model.state_dict(), os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' + 'best_metric_model.pth'))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
                            dice_val_best, dice_val
                        )
                    )
            global_step += 1
        return global_step, dice_val_best, global_step_best

    def forward(self):
        global_step = 0
        dice_val_best = 0.0
        global_step_best = 0
        while global_step < self.max_iterations:
            global_step, dice_val_best, global_step_best = self.train(global_step, dice_val_best, global_step_best)
        # self.model.load_state_dict(torch.load(os.path.join(self.defaults_path, 'weight/', self.model_name, '/', self.loss_function_name, "/best_metric_model.pth")))

# class TrainUNet(nn.Module):
#     def __init__(
#         self, 
#         model_name,
#         model,
#         loss_function_name,
#         loss_function,
#         max_epochs,
#         device,
#         val_interval,
#         optimizer_name,
#         defaults_path,
#         train_loader,
#         train_ds,
#         val_loader,
#         to_onehot
#     ) -> None:
#         super(TrainUNet, self).__init__()
#         # super().__init__()
#         self.model_name = model_name
#         self.model = model
#         self.loss_function_name = loss_function_name
#         self.loss_function = loss_function
#         self.max_epochs = max_epochs
#         self.device = device
#         self.val_interval = val_interval
#         self.optimizer_name = optimizer_name
#         self.defaults_path = defaults_path
#         self.train_loader = train_loader
#         self.train_ds = train_ds
#         self.val_loader = val_loader
#         self.to_onehot = to_onehot

#     def optimizer_list(self):
#         if self.optimizer_name == 'Adam':
#             optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
#         if self.optimizer_name == 'AdamW':
#             optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

#         return optimizer

#     def forward(self):

#         epoch_loss_values = []
#         metric_values = []

#         for epoch in range(self.max_epochs):
#             post_pred = Compose([AsDiscrete(argmax=True, to_onehot=self.to_onehot)])
#             post_label = Compose([AsDiscrete(to_onehot=self.to_onehot)])
#             dice_metric = DiceMetric(include_background=False, reduction="mean")
#             best_metric = -1
#             best_metric_epoch = -1

#             print("-" * 10)
#             print(f"epoch {epoch + 1}/{self.max_epochs}")
#             self.model.train()
#             epoch_loss = 0
#             step = 0
#             for batch_data in self.train_loader:
#                 step += 1
#                 inputs, labels = (
#                     batch_data["image"].to(self.device),
#                     batch_data["label"].to(self.device),
#                 )

#                 optimizer = self.optimizer_list()
#                 optimizer.zero_grad()

#                 outputs = self.model(inputs)
#                 loss = self.loss_function(outputs, labels)

#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
#                 print(f"{step}/{len(self.train_ds) // self.train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
#             epoch_loss /= step
#             epoch_loss_values.append(epoch_loss)
#             print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

#             if (epoch + 1) % self.val_interval == 0:
#                 self.model.eval()
#                 with torch.no_grad():
#                     for val_data in self.val_loader:
#                         val_inputs, val_labels = (
#                             val_data["image"].to(self.device),
#                             val_data["label"].to(self.device),
#                         )
#                         roi_size = (96, 96, 96)
#                         sw_batch_size = 4
#                         val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, self.model)
#                         val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
#                         val_labels = [post_label(i) for i in decollate_batch(val_labels)]
#                         # compute metric for current iteration
#                         dice_metric(y_pred=val_outputs, y=val_labels)

#                     # aggregate the final mean dice result
#                     metric = dice_metric.aggregate().item()
#                     # reset the status for next validation round
#                     dice_metric.reset()

#                     metric_values.append(metric)
#                     if metric > best_metric:
#                         best_metric = metric
#                         best_metric_epoch = epoch + 1

#                         if not os.path.exists(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/')):
#                             os.makedirs(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/'))

#                         torch.save(self.model.state_dict(), os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' + "best_metric_model.pth"))
#                         print("saved new best metric model")
#                     print(
#                         f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
#                         f"\nbest mean dice: {best_metric:.4f} "
#                         f"at epoch: {best_metric_epoch}"
#                     )

# class TrainUNETR(nn.Module):
#     def __init__(
#         self,
#         model_name,
#         model,
#         loss_function_name,
#         loss_function,
#         max_iterations,
#         eval_num,
#         optimizer_name,
#         defaults_path,
#         train_loader,
#         val_loader,
#         to_onehot
#     ) -> None:
#         super(TrainUNETR, self).__init__()
#         # super().__init__()
#         self.model_name = model_name
#         self.model = model
#         self.loss_function_name = loss_function_name
#         self.loss_function = loss_function
#         self.max_iterations = max_iterations
#         self.eval_num = eval_num
#         self.optimizer_name = optimizer_name
#         self.defaults_path = defaults_path
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.to_onehot = to_onehot

#     def optimizer_list(self):
#         if self.optimizer_name == 'Adam':
#             optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
#         if self.optimizer_name == 'AdamW':
#             optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

#         return optimizer

#     def validation(self, global_step, epoch_iterator_val):
#         post_label = AsDiscrete(to_onehot=self.to_onehot)
#         post_pred = AsDiscrete(argmax=True, to_onehot=self.to_onehot)
#         dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

#         self.model.eval()
#         with torch.no_grad():
#             for batch in epoch_iterator_val:
#                 val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
#                 val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model)
#                 val_labels_list = decollate_batch(val_labels)
#                 val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#                 val_outputs_list = decollate_batch(val_outputs)
#                 val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#                 dice_metric(y_pred=val_output_convert, y=val_labels_convert)
#                 epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
#             mean_dice_val = dice_metric.aggregate().item()
#             dice_metric.reset()
#         return mean_dice_val


#     def train(self,global_step, dice_val_best, global_step_best):
#         self.model.train()
#         epoch_loss = 0
#         step = 0
#         epoch_loss_values = []
#         metric_values = []
#         epoch_iterator = tqdm(self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
#         for step, batch in enumerate(epoch_iterator):
#             step += 1
#             x, y = (batch["image"].cuda(), batch["label"].cuda())
#             logit_map = self.model(x)
#             loss = self.loss_function(logit_map, y)
#             loss.backward()
#             epoch_loss += loss.item()
#             optimizer = self.optimizer_list()
#             optimizer.step()
#             optimizer.zero_grad()
#             epoch_iterator.set_description(  # noqa: B038
#                 "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, self.max_iterations, loss)
#             )
#             if (global_step % self.eval_num == 0 and global_step != 0) or global_step == self.max_iterations:
#                 epoch_iterator_val = tqdm(self.val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
#                 dice_val = self.validation(global_step, epoch_iterator_val)
#                 epoch_loss /= step
#                 epoch_loss_values.append(epoch_loss)
#                 metric_values.append(dice_val)
#                 if dice_val > dice_val_best:
#                     dice_val_best = dice_val
#                     global_step_best = global_step
#                     if not os.path.exists(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/')):
#                         os.makedirs(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' ))
#                     torch.save(self.model.state_dict(), os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' + 'best_metric_model.pth'))
#                     print(
#                         "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
#                     )
#                 else:
#                     print(
#                         "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
#                             dice_val_best, dice_val
#                         )
#                     )
#             global_step += 1
#         return global_step, dice_val_best, global_step_best

#     def forward(self):
#         global_step = 0
#         dice_val_best = 0.0
#         global_step_best = 0
#         while global_step < self.max_iterations:
#             global_step, dice_val_best, global_step_best = self.train(global_step, dice_val_best, global_step_best)
#         self.model.load_state_dict(torch.load(os.path.join(self.defaults_path, 'weight/UNETR_/', self.loss_function_name, "/best_metric_model.pth")))

# class TrainSwinUNETR(nn.Module):
#     def __init__(
#         self,
#         model_name,
#         model,
#         loss_function_name,
#         loss_function,
#         max_iterations,
#         eval_num,
#         optimizer_name,
#         defaults_path,
#         train_loader,
#         val_loader,
#         to_onehot
#     ) -> None:
#         super(TrainSwinUNETR, self).__init__()
#         # super().__init__()
#         self.model_name = model_name
#         self.model = model
#         self.loss_function_name = loss_function_name
#         self.loss_function = loss_function
#         self.max_iterations = max_iterations
#         self.eval_num = eval_num
#         self.optimizer_name = optimizer_name
#         self.defaults_path = defaults_path
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.to_onehot = to_onehot


#     def optimizer_list(self):
#         if self.optimizer_name == 'Adam':
#             optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
#         if self.optimizer_name == 'AdamW':
#             optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

#         return optimizer

#     def validation(self, global_step, epoch_iterator_val):
#         post_label = AsDiscrete(to_onehot= self.to_onehot)
#         post_pred = AsDiscrete(argmax=True, to_onehot= self.to_onehot)
#         dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

#         self.model.eval()
#         with torch.no_grad():
#             for batch in epoch_iterator_val:
#                 val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
#                 with torch.cuda.amp.autocast():
#                     val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model)
#                 val_labels_list = decollate_batch(val_labels)
#                 val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#                 val_outputs_list = decollate_batch(val_outputs)
#                 val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#                 dice_metric(y_pred=val_output_convert, y=val_labels_convert)
#                 epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
#             mean_dice_val = dice_metric.aggregate().item()
#             dice_metric.reset()
#         return mean_dice_val



#     def train(self, global_step, dice_val_best, global_step_best):
#         epoch_loss_values = []
#         metric_values = []
#         scaler = torch.cuda.amp.GradScaler()
#         self.model.train()
#         epoch_loss = 0
#         step = 0
#         epoch_iterator = tqdm(self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
#         for step, batch in enumerate(epoch_iterator):
#             step += 1
#             x, y = (batch["image"].cuda(), batch["label"].cuda())
#             with torch.cuda.amp.autocast():
#                 logit_map = self.model(x)
#                 loss = self.loss_function(logit_map, y)
#             scaler.scale(loss).backward()
#             epoch_loss += loss.item()

#             optimizer = self.optimizer_list()
#             scaler.unscale_(optimizer)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
#             epoch_iterator.set_description(  # noqa: B038
#                 f"Training ({global_step} / {self.max_iterations} Steps) (loss={loss:2.5f})"
#             )
#             if (global_step % self.eval_num == 0 and global_step != 0) or global_step == self.max_iterations:
#                 epoch_iterator_val = tqdm(self.val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
#                 dice_val = self.validation(global_step, epoch_iterator_val)
#                 epoch_loss /= step
#                 epoch_loss_values.append(epoch_loss)
#                 metric_values.append(dice_val)
#                 if dice_val > dice_val_best:
#                     dice_val_best = dice_val
#                     global_step_best = global_step
#                     if not os.path.exists(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/')):
#                         os.makedirs(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/'))
#                     torch.save(self.model.state_dict(), os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' + '/best_metric_model.pth'))
#                     print(
#                         "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
#                     )
#                 else:
#                     print(
#                         "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
#                             dice_val_best, dice_val
#                         )
#                     )
#             global_step += 1
#         return global_step, dice_val_best, global_step_best

#     def forward(self):
#         global_step = 0
#         dice_val_best = 0.0
#         global_step_best = 0
#         while global_step < self.max_iterations:
#             global_step, dice_val_best, global_step_best = self.train(global_step, dice_val_best, global_step_best)

                    
# class TrainMultiHeaderUNet(nn.Module):
#     def __init__(
#         self, 
#         model_name,
#         model,
#         loss_function_name,
#         loss_function,
#         distance_map_weight,
#         max_epochs,
#         device,
#         val_interval,
#         optimizer_name,
#         defaults_path,
#         train_loader,
#         train_ds,
#         val_loader,
#         to_onehot
#     ) -> None:
#         super(TrainMultiHeaderUNet, self).__init__()
#         # super().__init__()
#         self.model_name = model_name
#         self.model = model
#         self.loss_function_name = loss_function_name
#         self.loss_function = loss_function
#         self.distance_map_weight = distance_map_weight
#         self.max_epochs = max_epochs
#         self.device = device
#         self.val_interval = val_interval
#         self.optimizer_name = optimizer_name
#         self.defaults_path = defaults_path
#         self.train_loader = train_loader
#         self.train_ds = train_ds
#         self.val_loader = val_loader
#         self.to_onehot = to_onehot

#     def optimizer_list(self):
#         if self.optimizer_name == 'Adam':
#             optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
#         if self.optimizer_name == 'AdamW':
#             optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

#         return optimizer

#     def forward(self):
#         epoch_loss_values = []
#         metric_values = []

#         for epoch in range(self.max_epochs):
#             post_pred = Compose([AsDiscrete(argmax=True, to_onehot=self.to_onehot)])
#             post_label = Compose([AsDiscrete(to_onehot=self.to_onehot)])
#             dice_metric = DiceMetric(include_background=False, reduction="mean")
#             best_metric = -1
#             best_metric_epoch = -1

#             print("-" * 10)
#             print(f"epoch {epoch + 1}/{self.max_epochs}")
#             self.model.train()
#             epoch_loss = 0
#             step = 0
#             for batch_data in self.train_loader:
#                 step += 1
#                 inputs, labels = (
#                     batch_data["image"].to(self.device),
#                     batch_data["label"].to(self.device),
#                 )

#                 optimizer = self.optimizer_list()
#                 optimizer.zero_grad()
# ########################################################################################################################################################################
#                 outputs, outputs_dist = self.model(inputs)
#                 outputs_dist = nn.Softplus()(outputs_dist)
#                 loss = self.loss_function(outputs, outputs_dist, labels, self.distance_map_weight)
# ########################################################################################################################################################################
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
#                 print(f"{step}/{len(self.train_ds) // self.train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
#             epoch_loss /= step
#             epoch_loss_values.append(epoch_loss)
#             print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

#             if (epoch + 1) % self.val_interval == 0:
#                 self.model.eval()
#                 with torch.no_grad():
#                     for val_data in self.val_loader:
#                         val_inputs, val_labels = (
#                             val_data["image"].to(self.device),
#                             val_data["label"].to(self.device),
#                         )
#                         roi_size = (96, 96, 96)
#                         sw_batch_size = 4
#                         val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, self.model)
#                         val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
#                         val_labels = [post_label(i) for i in decollate_batch(val_labels)]

#                         # compute metric for current iteration
#                         # print(type(val_outputs))
#                         # print(type(val_labels))

#                         # val_outputs = torch.stack(val_outputs)
#                         # val_labels = torch.stack(val_labels)

#                         dice_metric(y_pred=val_outputs, y=val_labels)

#                     # aggregate the final mean dice result
#                     metric = dice_metric.aggregate().item()
#                     # reset the status for next validation round
#                     dice_metric.reset()

#                     metric_values.append(metric)
#                     if metric > best_metric:
#                         best_metric = metric
#                         best_metric_epoch = epoch + 1

#                         if not os.path.exists(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/')):
#                             os.makedirs(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/'))

#                         torch.save(self.model.state_dict(), os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' + "best_metric_model.pth"))
#                         print("saved new best metric model")
#                     print(
#                         f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
#                         f"\nbest mean dice: {best_metric:.4f} "
#                         f"at epoch: {best_metric_epoch}"
#                     )


# class TrainMultiHeaderSwinUNETR(nn.Module):
#     def __init__(
#         self,
#         model_name,
#         model,
#         loss_function_name,
#         loss_function,
#         distance_map_weight,
#         max_iterations,
#         eval_num,
#         optimizer_name,
#         defaults_path,
#         train_loader,
#         val_loader,
#         to_onehot
#     ) -> None:
#         super(TrainMultiHeaderSwinUNETR, self).__init__()
#         # super().__init__()
#         self.model_name = model_name
#         self.model = model
#         self.loss_function_name = loss_function_name
#         self.loss_function = loss_function
#         self.distance_map_weight = distance_map_weight
#         self.max_iterations = max_iterations
#         self.eval_num = eval_num
#         self.optimizer_name = optimizer_name
#         self.defaults_path = defaults_path
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.to_onehot = to_onehot


#     def optimizer_list(self):
#         if self.optimizer_name == 'Adam':
#             optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
#         if self.optimizer_name == 'AdamW':
#             optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

#         return optimizer

#     def validation(self, global_step, epoch_iterator_val):
#         post_label = AsDiscrete(to_onehot= self.to_onehot)
#         post_pred = AsDiscrete(argmax=True, to_onehot= self.to_onehot)
#         dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

#         self.model.eval()
#         with torch.no_grad():
#             for batch in epoch_iterator_val:
#                 val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
#                 with torch.cuda.amp.autocast():
#                     val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model)
#                 val_labels_list = decollate_batch(val_labels)
#                 val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#                 val_outputs_list = decollate_batch(val_outputs)
#                 val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#                 dice_metric(y_pred=val_output_convert, y=val_labels_convert)
#                 epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
#             mean_dice_val = dice_metric.aggregate().item()
#             dice_metric.reset()
#         return mean_dice_val



#     def train(self, global_step, dice_val_best, global_step_best):
#         epoch_loss_values = []
#         metric_values = []
#         scaler = torch.cuda.amp.GradScaler()
#         self.model.train()
#         epoch_loss = 0
#         step = 0
#         epoch_iterator = tqdm(self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
#         for step, batch in enumerate(epoch_iterator):
#             step += 1
#             x, y = (batch["image"].cuda(), batch["label"].cuda())
#             with torch.cuda.amp.autocast():
#                 # logit_map = self.model(x)
# ########################################################################################################################################################################
#                 outputs, outputs_dist = self.model(x)
#                 outputs_dist = nn.Softplus()(outputs_dist)
#                 loss = self.loss_function(outputs, outputs_dist, y, self.distance_map_weight)
# ########################################################################################################################################################################

#             scaler.scale(loss).backward()
#             epoch_loss += loss.item()

#             optimizer = self.optimizer_list()
#             scaler.unscale_(optimizer)
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()
#             epoch_iterator.set_description(  # noqa: B038
#                 f"Training ({global_step} / {self.max_iterations} Steps) (loss={loss:2.5f})"
#             )
#             if (global_step % self.eval_num == 0 and global_step != 0) or global_step == self.max_iterations:
#                 epoch_iterator_val = tqdm(self.val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
#                 dice_val = self.validation(global_step, epoch_iterator_val)
#                 epoch_loss /= step
#                 epoch_loss_values.append(epoch_loss)
#                 metric_values.append(dice_val)
#                 if dice_val > dice_val_best:
#                     dice_val_best = dice_val
#                     global_step_best = global_step
#                     if not os.path.exists(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/')):
#                         os.makedirs(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/'))
#                     torch.save(self.model.state_dict(), os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' + '/best_metric_model.pth'))
#                     print(
#                         "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
#                     )
#                 else:
#                     print(
#                         "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
#                             dice_val_best, dice_val
#                         )
#                     )
#             global_step += 1
#         return global_step, dice_val_best, global_step_best

#     def forward(self):
#         global_step = 0
#         dice_val_best = 0.0
#         global_step_best = 0
#         while global_step < self.max_iterations:
#             global_step, dice_val_best, global_step_best = self.train(global_step, dice_val_best, global_step_best)
            
# ####################################################################################Temp####################################################################################
# ####################################################################################Temp####################################################################################
# ####################################################################################Temp####################################################################################

# class TrainUNETR2(nn.Module):
#     def __init__(
#         self,
#         model_name,
#         model,
#         loss_function_name,
#         loss_function,
#         max_iterations,
#         eval_num,
#         optimizer_name,
#         defaults_path,
#         train_loader,
#         val_loader,
#         to_onehot
#     ) -> None:
#         super(TrainUNETR2, self).__init__()
#         # super().__init__()
#         self.model_name = model_name
#         self.model = model
#         self.loss_function_name = loss_function_name
#         self.loss_function = loss_function
#         self.max_iterations = max_iterations
#         self.eval_num = eval_num
#         self.optimizer_name = optimizer_name
#         self.defaults_path = defaults_path
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.to_onehot = to_onehot

#     def optimizer_list(self):
#         if self.optimizer_name == 'Adam':
#             optimizer = torch.optim.Adam(self.model.parameters(), 1e-4)
#         if self.optimizer_name == 'AdamW':
#             optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)

#         return optimizer

#     def validation(self, global_step, epoch_iterator_val):
#         post_label = AsDiscrete(to_onehot=self.to_onehot)
#         post_pred = AsDiscrete(argmax=True, to_onehot=self.to_onehot)
#         dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

#         self.model.eval()
#         with torch.no_grad():
#             for batch in epoch_iterator_val:
#                 val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
#                 val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, self.model)
#                 val_labels_list = decollate_batch(val_labels)
#                 val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#                 val_outputs_list = decollate_batch(val_outputs)
#                 val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#                 dice_metric(y_pred=val_output_convert, y=val_labels_convert)
#                 epoch_iterator_val.set_description("Validate (%d / %d Steps)" % (global_step, 10.0))  # noqa: B038
#             mean_dice_val = dice_metric.aggregate().item()
#             dice_metric.reset()
#         return mean_dice_val


#     def train(self,global_step, dice_val_best, global_step_best):
#         self.model.train()
#         epoch_loss = 0
#         step = 0
#         epoch_loss_values = []
#         metric_values = []
#         epoch_iterator = tqdm(self.train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
#         for step, batch in enumerate(epoch_iterator):
#             step += 1
#             x, y = (batch["image"].cuda(), batch["label"].cuda())
#             logit_map = self.model(x)
#             loss = self.loss_function(logit_map, y)
#             loss.backward()
#             epoch_loss += loss.item()
#             optimizer = self.optimizer_list()
#             optimizer.step()
#             optimizer.zero_grad()
#             epoch_iterator.set_description(  # noqa: B038
#                 "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, self.max_iterations, loss)
#             )
#             if (global_step % self.eval_num == 0 and global_step != 0) or global_step == self.max_iterations:
#                 epoch_iterator_val = tqdm(self.val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
#                 dice_val = self.validation(global_step, epoch_iterator_val)
#                 epoch_loss /= step
#                 epoch_loss_values.append(epoch_loss)
#                 metric_values.append(dice_val)
#                 if dice_val > dice_val_best:
#                     dice_val_best = dice_val
#                     global_step_best = global_step
#                     if not os.path.exists(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/')):
#                         os.makedirs(os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' ))
#                     torch.save(self.model.state_dict(), os.path.join(self.defaults_path + 'weight/'+ self.model_name + '_' + self.loss_function_name + '/' + 'best_metric_model_temp.pth'))
#                     print(
#                         "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
#                     )
#                 else:
#                     print(
#                         "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(
#                             dice_val_best, dice_val
#                         )
#                     )
#             global_step += 1
#         return global_step, dice_val_best, global_step_best

#     def forward(self):
#         global_step = 0
#         dice_val_best = 0.0
#         global_step_best = 0
#         while global_step < self.max_iterations:
#             global_step, dice_val_best, global_step_best = self.train(global_step, dice_val_best, global_step_best)
#         self.model.load_state_dict(torch.load(os.path.join(self.defaults_path, 'weight/UNETR_/', self.loss_function_name, "/best_metric_model.pth")))
