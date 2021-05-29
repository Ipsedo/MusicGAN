import torch as th
import torch.nn as nn


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        norm = th.sqrt((x ** 2).sum(dim=1))
        return x / norm.unsqueeze(1)


class TransConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
    ):
        super(TransConvBlock, self).__init__()

        self.__tr_conv_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                stride=(2, 2),
                kernel_size=(4, 4),
                padding=(1, 1)
            ),
            nn.LeakyReLU(1e-1),
            PixelNorm()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__tr_conv_block(x)


class GenBlock(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(GenBlock, self).__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.LeakyReLU(1e-1),
            PixelNorm(),
            nn.Upsample(
                scale_factor=2,
                mode="bilinear"
            )
        )


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            out_channel: int
    ):
        super(Generator, self).__init__()

        nb_layer = 8

        channel_list = [
            (rand_channels, 256),
            (256, 224),
            (224, 192),
            (192, 160),
            (160, 128),
            (128, 96),
            (96, 64),
            (64, 32)
        ]

        self.__gen = nn.Sequential(*[
            TransConvBlock(
                channel_list[i][0],
                channel_list[i][1]
            )
            for i in range(nb_layer)
        ])

        self.__conv_out = nn.Sequential(
            nn.Conv2d(
                channel_list[-1][1],
                out_channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.Tanh()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__gen(x)

        out = self.__conv_out(out)

        return out
