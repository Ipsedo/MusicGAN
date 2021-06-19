import torch as th
import torch.nn as nn


class NoiseLayer(nn.Module):
    def __init__(
            self,
            channels: int
    ):
        super(NoiseLayer, self).__init__()

        self.__weights = nn.Parameter(
            th.zeros(1, channels, 1, 1)
        )

    def forward(
            self,
            x: th.Tensor
    ) -> th.Tensor:
        noise = th.randn(
            x.size()[0], 1,
            x.size()[2], x.size()[3],
            device=x.device
        )

        out = x + self.__weights * noise

        return out

    def __repr__(self):
        return f"NoiseLayer" \
               f"({self.__weights.size()[1]})"

    def __str__(self):
        return self.__repr__()


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
            )
        )

        if last_layer:
            self.__block.add_module(
                "2", nn.Tanh()
            )
        else:
            self.__block.add_module(
                "2", nn.InstanceNorm2d(
                    out_channels,
                    affine=True
                )
            )

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
