import torch as th
import torch.nn as nn


class PixelNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super(PixelNorm, self).__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        norm = th.sqrt(
            x.pow(2.).mean(dim=1, keepdim=True) +
            self.__epsilon
        )

        return x / norm


class AdaIN(nn.Module):
    def __init__(
            self,
            channels: int,
            style_channels: int
    ):
        super(AdaIN, self).__init__()

        self.__to_style = nn.Linear(
            style_channels,
            2 * channels,
            bias=False
        )

        self.__inst_norm = nn.InstanceNorm2d(
            channels, affine=False
        )

        self.__channels = channels
        self.__style_channels = style_channels

    def forward(self, x: th.Tensor, z: th.Tensor) -> th.Tensor:
        b, c, _, _ = x.size()

        out_lin = self.__to_style(z).view(b, 2 * c, 1, 1)
        gamma, beta = out_lin.chunk(2, 1)

        out_norm = self.__inst_norm(x)

        out = gamma * out_norm + beta

        return out

    def __repr__(self):
        return "AdaIN" + \
               f"(channels={self.__channels}, " \
               f"style={self.__style_channels})"

    def __str__(self):
        return self.__repr__()


class GenConv2d(nn.ConvTranspose2d):
    def __init__(self, in_channels: int, out_channels: int):
        super(GenConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1)
        )


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int
    ):
        super(Block, self).__init__()

        self.__conv = GenConv2d(
            in_channels,
            out_channels
        )

        self.__lrelu = nn.LeakyReLU(2e-1)

        self.__adain = AdaIN(
            out_channels, style_channels
        )

    def forward(
            self,
            x: th.Tensor,
            style: th.Tensor
    ) -> th.Tensor:
        out_conv = self.__conv(x)

        out_adain = self.__adain(out_conv, style)

        out_lrelu = self.__lrelu(out_adain)

        return out_lrelu


class EndBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(EndBlock, self).__init__()

        self.__block = nn.Sequential(
            GenConv2d(
                in_channels,
                out_channels
            ),
            nn.Tanh()
        )

    def forward(
            self,
            x: th.Tensor,
            unused=None
    ) -> th.Tensor:
        out = self.__block(x)

        return out


class LinearBlock(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(LinearBlock, self).__init__(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(2e-1)
        )


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            style_channels: int
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
                style_channels
            ) if i != len(channels) - 1
            else EndBlock(
                c[0], c[1]
            )
            for i, c in enumerate(channels)
        ])

        self.__style_network = nn.Sequential(*[
            LinearBlock(style_channels, style_channels)
            for _ in range(4)
        ])

    def forward(
            self,
            z: th.Tensor,
            z_style: th.Tensor
    ) -> th.Tensor:
        style = self.__style_network(z_style)

        out = z

        for m in self.__gen_blocks:
            out = m(out, style)

        return out
