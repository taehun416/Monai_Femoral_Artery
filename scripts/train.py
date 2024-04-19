import autorootcwd
from src.network.base_network import ModelRegistry
from torchsummary import summary

network = ModelRegistry.create(
    "UNet",
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64),
    strides=(2, 2, 2),
    num_res_units=2,
)

summary(network, ((1, 32, 32, 32)), depth=5)
