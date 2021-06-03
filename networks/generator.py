import torch as th
import torch.nn as nn


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        norm = th.sqrt(th.pow(x, 2.).mean(dim=1))
        return x / norm.unsqueeze(1)


class TransConvBlock(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            last_layer: bool
    ):
        super(TransConvBlock, self).__init__(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=(1, 1),
                output_padding=(1, 1)
            )
        )

        if last_layer:
            self.add_module("1", nn.Tanh())
        else:
            self.add_module("1", nn.LeakyReLU(2e-1))
            self.add_module("2", nn.BatchNorm2d(out_channels))


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            out_channel: int
    ):
        super(Generator, self).__init__()

        nb_layer = 9

        channel_list = [
            (rand_channels, 256),
            (256, 224),
            (224, 192),
            (192, 160),
            (160, 128),
            (128, 96),
            (96, 64),
            (64, 32),
            (32, out_channel)
        ]

        self.__gen = nn.Sequential(*[
            TransConvBlock(
                channel_list[i][0],
                channel_list[i][1],
                i == nb_layer - 1
            )
            for i in range(nb_layer)
        ])

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__gen(x)

        return out
