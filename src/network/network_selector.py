from src.network import UNet, UNETR, SwinUNETR, UnetImageCAS
from src.network.multi_head_3D_unet import MultiHeadUNet
from monai.networks.layers import Norm
from src.network.multiheader_swin_unetr import MultiHeaderSwinUNETR


class NetworkSelector:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

    def get_model(self):
        if self.model_name == 'UNet':
            return self.get_unet()
        elif self.model_name == 'UNet_dist':
            return self.get_multihead_unet()
        elif self.model_name == 'UNETR':
            return self.get_unetr()
        elif self.model_name == 'UNETR2':
            return self.get_unetr()
        elif self.model_name == 'UNet_Multiheader':
            return self.get_unet_imagecas()   
        elif self.model_name == 'SwinUNETR_Multiheader':
            return self.get_multiheader_swinunetr()   
        elif self.model_name == ('SwinUNETR') or ('SwinUNETR_Cropsize') or ('SwinUNETR_Cropsize_Rot') or ('SwinUNETR_Cropsize_RandFlip') or ('SwinUNETR_Cropsize_RandShiftIntensityd') or ('SwinUNETR_All') or ('SwinUNETR_All2'):
            return self.get_swinunetr() 
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
        
    def get_unet_imagecas(self):
        return UnetImageCAS(3, 1).to(self.device)


    def get_unet(self):
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).to(self.device)

    # def get_multihead_unet(self):
    #     return MultiHeadUNet(
    #         spatial_dims=3,
    #         in_channels=1,
    #         out_channels=1,
    #         channels=(16, 32, 64, 128, 256),
    #         strides=(2, 2, 2, 2),
    #         num_res_units=2,
    #         norm=Norm.BATCH,
    #     ).to(self.device)

    def get_unetr(self):
        return UNETR(
            in_channels=1,
            out_channels=2,
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
        ).to(self.device)

    def get_swinunetr(self):
        return SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        ).to(self.device)

    def get_multiheader_swinunetr(self):
        return MultiHeaderSwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=2,
            feature_size=48,
            use_checkpoint=True,
        ).to(self.device)