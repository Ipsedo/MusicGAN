import torch as th
import torch.nn as nn


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            last_layer: bool
    ):
        super(Block, self).__init__()

        self.__block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            nn.BatchNorm2d(out_channels)
        )

        if last_layer:
            self.__block.add_module(
                "3", nn.Tanh()
            )
        else:
            self.__block.add_module(
                "3", nn.LeakyReLU(2e-1)
            )

    def forward(
            self,
            x: th.Tensor,
    ) -> th.Tensor:
        out = self.__block(x)

        return out


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int
    ):
        super(Generator, self).__init__()

        channels = [
            (rand_channels, 224),
            (224, 192),
            (192, 160),
            (160, 128),
            (128, 96),
            (96, 64),
            (64, 32),
            (32, 2)
        ]

        # Generator layers
        self.__gen_blocks = nn.Sequential(*[
            Block(
                c[0], c[1],
                i == len(channels) - 1
            )
            for i, c in enumerate(channels)
        ])

    def forward(
            self,
            z: th.Tensor
    ) -> th.Tensor:
        out = self.__gen_blocks(z)

        return out
