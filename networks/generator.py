import torch as th
import torch.nn as nn

from typing import Tuple


class ConstantLayer(nn.Module):
    def __init__(
            self,
            channels: int,
            sizes: Tuple[int, int]
    ):
        super(ConstantLayer, self).__init__()

        self.__constant = nn.Parameter(
            th.randn(1, channels, sizes[0], sizes[1])
        )

    def forward(self, batch_size: int) -> th.Tensor:
        constant = self.__constant.repeat(batch_size, 1, 1, 1)

        return constant

    def __repr__(self):
        return f"ConstantLayer(" \
               f"{self.__constant.size()[1]}, " \
               f"size=({self.__constant.size()[2]}, " \
               f"{self.__constant.size()[3]}))"

    def __str__(self):
        return self.__repr__()


class AdaptiveInstanceNorm(nn.Module):
    def __init__(
            self,
            in_channels: int,
            style_channels: int
    ):
        super(AdaptiveInstanceNorm, self).__init__()

        self.__style = nn.Linear(style_channels, in_channels * 2)

    def forward(
            self,
            x: th.Tensor,
            y: th.Tensor
    ) -> th.Tensor:
        batch_size = x.size()[0]
        in_channels = x.size()[1]

        # (Nb, 2 * Nc, 1, 1)
        style = self.__style(y) \
            .view(batch_size, -1, 1, 1)

        mean = x.view(batch_size, in_channels, -1) \
            .mean(dim=2) \
            .view(batch_size, in_channels, 1, 1)

        std = x.view(batch_size, in_channels, -1) \
            .std(dim=2) \
            .view(batch_size, in_channels, 1, 1)

        y_b, y_s = style.chunk(2, 1)

        x_norm = (x - mean) / std

        out = x_norm * y_s + y_b

        return out

    def __repr__(self):
        return f"AdaptiveInstanceNorm(" \
               f"style_channels={self.__style.weight.size()[1]}, " \
               f"in_channels={self.__style.weight.size()[0] // 2})"

    def __str__(self):
        return self.__repr__()


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
            x.size()[0], 1, x.size()[2], x.size()[3],
            device=x.device
        )

        out = x + self.__weights * noise

        return out

    def __repr__(self):
        return f"NoiseLayer({self.__weights.size()[1]})"

    def __str__(self):
        return self.__repr__()


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int
    ):
        super(Block, self).__init__()

        self.__conv_tr = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            stride=(2, 2),
            kernel_size=(3, 3),
            padding=(1, 1),
            output_padding=(1, 1)
        )

        self.__noise = NoiseLayer(
            out_channels
        )

        self.__adain = AdaptiveInstanceNorm(
            out_channels, style_channels
        )

    def forward(
            self,
            x: th.Tensor,
            style: th.Tensor
    ) -> th.Tensor:
        out_conv = self.__conv_tr(x)
        out_noise = self.__noise(out_conv)
        out_adain = self.__adain(out_noise, style)

        return out_adain


class InputBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            in_sizes: Tuple[int, int],
            style_channels: int
    ):
        super(InputBlock, self).__init__()

        self.__input = ConstantLayer(
            in_channels,
            in_sizes
        )

        self.__noise = NoiseLayer(
            in_channels
        )

        self.__adain = AdaptiveInstanceNorm(
            in_channels, style_channels
        )

    def forward(
            self,
            style: th.Tensor,
            nb_input: int = 1
    ) -> th.Tensor:
        batch_size = style.size()[0]

        out_input = th.cat([
            self.__input(batch_size)
            for _ in range(nb_input)
        ], dim=2)

        out_noise = self.__noise(out_input)
        out_adain = self.__adain(out_noise, style)

        return out_adain


class Generator(nn.Module):
    def __init__(
            self,
            style_channels: int
    ):
        super(Generator, self).__init__()

        channels = [
            (256, 224),
            (224, 192),
            (192, 160),
            (160, 128),
            (128, 96),
            (96, 64),
            (64, 32)
        ]

        in_sizes = (2, 2)

        # Input layer
        self.__input_block = InputBlock(
            channels[0][0],
            in_sizes,
            style_channels
        )

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            Block(
                c[0], c[1],
                style_channels
            )
            for c in channels
        ])

        # Output layer
        self.__last_block = nn.Sequential(
            nn.ConvTranspose2d(
                channels[-1][1], 2,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            nn.Tanh()
        )

        # Style network
        tmp_style_layers = []

        for _ in range(4):
            tmp_style_layers.append(
                nn.Linear(
                    style_channels,
                    style_channels
                )
            )
            tmp_style_layers.append(
                nn.LeakyReLU(2e-1)
            )

        self.__style_layers = nn.Sequential(
            *tmp_style_layers
        )

    def forward(
            self,
            z: th.Tensor,
            nb_input: int = 1
    ) -> th.Tensor:
        style = self.__style_layers(z)

        out = self.__input_block(style, nb_input)

        for gen_block in self.__gen_blocks:
            out = gen_block(out, style)

        out = self.__last_block(out)

        return out
