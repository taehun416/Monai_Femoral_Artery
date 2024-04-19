
import torch.nn as nn
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandFlipd,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    RandRotated
)

class TrainTransform(nn.Module):
    def __init__(self) -> None:
        super(TrainTransform, self).__init__()

    @staticmethod
    def transform():
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024,
                    a_max=3071,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                # RandFlipd(
                #     keys=["image", "label"],
                #     spatial_axis=[0],
                #     prob=0.10,
                # ),
                # RandFlipd(
                #     keys=["image", "label"],
                #     spatial_axis=[1],
                #     prob=0.10,
                # ),
                # RandFlipd(
                #     keys=["image", "label"],
                #     spatial_axis=[2],
                #     prob=0.10,
                # ),
                # RandRotate90d(
                #     keys=["image", "label"],
                #     prob=0.10,
                #     max_k=3,
                # ),
                # RandShiftIntensityd(
                #     keys=["image"],
                #     offsets=0.10,
                #     prob=0.50,
                # ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], 
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return train_transforms, val_transforms
    

class TestTransform(nn.Module):
    def __init__(self) -> None:
        super(TestTransform, self).__init__()

    @staticmethod
    def transform():
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], 
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return test_transforms
    

######################################################################Aug_test######################################################################
######################################################################Aug_test######################################################################
######################################################################Aug_test######################################################################

class TrainTransformCropSize(nn.Module):
    def __init__(self) -> None:
        super(TrainTransformCropSize, self).__init__()

    @staticmethod
    def transform():
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(64, 64, 128),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], 
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return train_transforms, val_transforms

class TrainTransformCropSizeRot(nn.Module):
    def __init__(self) -> None:
        super(TrainTransformCropSizeRot, self).__init__()

    @staticmethod
    def transform():
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(64, 64, 128),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandRotated(
                    keys=["image", "label"],
                    prob=0.10,
                    range_x=0.4, 
                    range_y=0.4, 
                    range_z=0.4
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], 
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return train_transforms, val_transforms
    

class TrainTransformCropSizeRandFlip(nn.Module):
    def __init__(self) -> None:
        super(TrainTransformCropSizeRandFlip, self).__init__()

    @staticmethod
    def transform():
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(64, 64, 128),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], 
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return train_transforms, val_transforms

  

class TrainTransformCropSizeRandShiftIntensityd(nn.Module):
    def __init__(self) -> None:
        super(TrainTransformCropSizeRandShiftIntensityd, self).__init__()

    @staticmethod
    def transform():
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(64, 64, 128),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], 
                    a_min=-1024, 
                    a_max=3071, 
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return train_transforms, val_transforms
    

class TrainTransformAll(nn.Module):
    def __init__(self) -> None:
        super(TrainTransformAll, self).__init__()

    @staticmethod
    def transform():
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024,
                    a_max=3071,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(64, 64, 128),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),

                RandRotated(
                    keys=["image", "label"],
                    prob=0.10,
                    range_x=0.4, 
                    range_y=0.4, 
                    range_z=0.4
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10,
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], 
                    a_min=-1024,
                    a_max=3071,
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return train_transforms, val_transforms
    
class TrainTransformAll2(nn.Module):
    def __init__(self) -> None:
        super(TrainTransformAll, self).__init__()

    @staticmethod
    def transform():
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-1024,
                    a_max=3071,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(64, 64, 128),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),

                RandRotated(
                    keys=["image", "label"],
                    prob=0.10,
                    range_x=0.4, 
                    range_y=0.4, 
                    range_z=0.4
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], 
                    a_min=-1024,
                    a_max=3071,
                    b_min=0.0, 
                    b_max=1.0, 
                    clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
            ]
        )
        return train_transforms, val_transforms