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


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x: th.Tensor) -> th.Tensor:
        norm = th.sqrt(th.pow(x, 2.).mean(dim=1))
        return x / norm.unsqueeze(1)


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
        # (Nb, 2 * Nc, 1, 1)
        style = self.__style(y) \
            .unsqueeze(2) \
            .unsqueeze(2)

        in_channels = x.size()[1]

        beta, gamma = style[:, :in_channels], \
                      style[:, in_channels:]

        out = x * gamma + beta

        return out


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
        device = "cuda" if x.is_cuda else "cpu"

        noise = th.randn(x.size()[0], 1, x.size()[2], x.size()[3],
                         device=device)

        out = x + self.__weights * noise

        return out


class Conv2D3x3(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Conv2D3x3, self).__init__()

        self.__conv = nn.Conv2d(
            in_channels,
            out_channels,
            (3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

    def forward(
            self, x: th.Tensor
    ) -> th.Tensor:
        out = self.__conv(x)

        return out


class GeneratorBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            style_channels: int,
            last_layer: bool
    ):
        super(GeneratorBlock, self).__init__()

        self.__up_sample = nn.Upsample(scale_factor=2)

        self.__conv_1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            (3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.__noise_1 = NoiseLayer(
            hidden_channels
        )

        self.__adain_1 = AdaptiveInstanceNorm(
            hidden_channels, style_channels
        )

        self.__conv_2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            (3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.__noise_2 = NoiseLayer(out_channels)

        self.__adain_2 = AdaptiveInstanceNorm(
            out_channels, style_channels
        )

    def forward(
            self,
            x: th.Tensor,
            style: th.Tensor
    ) -> th.Tensor:
        out_up_sample = self.__up_sample(x)

        out_conv_1 = self.__conv_1(out_up_sample)
        out_noise_1 = self.__noise_1(out_conv_1)
        out_adain_1 = self.__adain_1(out_noise_1, style)

        out_conv_2 = self.__conv_2(out_adain_1)
        out_noise_2 = self.__noise_2(out_conv_2)
        out_adain_2 = self.__adain_2(out_noise_2, style)

        return out_adain_2


class InputBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            in_sizes: Tuple[int, int],
            style_channels: int
    ):
        super(InputBlock, self).__init__()

        self.__input = ConstantLayer(
            in_channels,
            in_sizes
        )

        self.__noise_1 = NoiseLayer(
            in_channels
        )

        self.__adain_1 = AdaptiveInstanceNorm(
            in_channels, style_channels
        )

        self.__conv = nn.Conv2d(
            in_channels,
            out_channels,
            (3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.__noise_2 = NoiseLayer(out_channels)

        self.__adain_2 = AdaptiveInstanceNorm(
            out_channels, style_channels
        )

    def forward(self, style: th.Tensor, nb_input: int = 1) -> th.Tensor:
        batch_size = style.size()[0]

        out_input = th.cat([
            self.__input(batch_size) for _ in range(nb_input)], dim=2)

        out_noise_1 = self.__noise_1(out_input)
        out_adain_1 = self.__adain_1(out_noise_1, style)

        out_conv = self.__conv(out_adain_1)
        out_noise_2 = self.__noise_2(out_conv)
        out_adain_2 = self.__adain_2(out_noise_2, style)

        return out_adain_2


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
        in_sizes = (4, 4)

        self.__input = InputBlock(
            channels[0][0],
            channels[0][1],
            in_sizes,
            style_channels
        )

        self.__gen_blocks = nn.ModuleList([
            GeneratorBlock(
                c[0], c[0], c[1],
                style_channels,
                i == len(channels) - 1
            )
            for i, c in enumerate(channels[1:])
        ])

        self.__style_layers = nn.Sequential(*[
            nn.Linear(style_channels, style_channels)
            for _ in range(4)
        ])

        self.__last_layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                channels[-1][1], 2,
                (3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.Tanh()
        )

    def forward(self, style: th.Tensor, nb_input: int = 1) -> th.Tensor:
        style = self.__style_layers(style)

        out = self.__input(style, nb_input)

        for gen_block in self.__gen_blocks:
            out = gen_block(out, style)

        out = self.__last_layers(out)

        return out
