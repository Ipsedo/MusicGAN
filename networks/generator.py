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

    def __repr__(self):
        return f"PixelNorm(eps={self.__epsilon})"

    def __str__(self):
        return self.__repr__()


class NoiseLayer(nn.Module):
    def __init__(self, channels: int):
        super(NoiseLayer, self).__init__()

        self.__channels = channels

        self.__to_noise = nn.Linear(1, channels, bias=False)

    def forward(self, x: th.Tensor) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        b, c, w, h = x.size()

        rand_per_pixel = th.randn(b, w, h, 1, device=device)

        out = x + self.__to_noise(rand_per_pixel).permute(0, 3, 1, 2)

        return out

    def __repr__(self):
        return f"NoiseLayer({self.__channels})"

    def __str__(self):
        return self.__repr__()


class AdaIN(nn.Module):
    def __init__(
            self,
            channels: int,
            style_channels: int
    ):
        super(AdaIN, self).__init__()

        self.__to_style = nn.Linear(
            style_channels,
            2 * channels
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

        self.__pn = PixelNorm()

        self.__adain = AdaIN(
            out_channels, style_channels
        )

        self.__lrelu = nn.LeakyReLU(2e-1)

    def forward(
            self,
            x: th.Tensor,
            style: th.Tensor
    ) -> th.Tensor:
        out = self.__conv(x)
        out = self.__pn(out)

        out = self.__adain(out, style)

        out = self.__lrelu(out)

        return out


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


class ToMagnPhaseLayer(nn.Sequential):
    def __init__(self, in_channels: int):
        super(ToMagnPhaseLayer, self).__init__(
            nn.Conv2d(
                in_channels, 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.Tanh()
        )


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            style_channels: int,
            start_layer: int = 0
    ):
        super(Generator, self).__init__()

        self.__start_layer = start_layer

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

        assert 0 <= start_layer < len(channels)

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            Block(
                c[0], c[1],
                style_channels
            ) if i != len(channels) - 1
            else EndBlock(
                c[0], c[1]
            )
            for i, c in enumerate(channels)
        ])

        # for progressive gan
        self.__end_blocks = nn.ModuleList([
            ToMagnPhaseLayer(c[0])
            for c in channels[1:]
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

        for i, m in enumerate(self.__gen_blocks):
            out = m(out, style)

            if i >= self.__start_layer:
                break

        if self.__start_layer < len(self.__gen_blocks) - 1:
            out = self.__end_blocks[self.__start_layer](out)

        return out

    def next_layer(self) -> None:
        self.__start_layer += 1

        if self.__start_layer >= len(self.__gen_blocks):
            self.__start_layer = len(self.__gen_blocks) - 1

    @property
    def nb_layer(self) -> int:
        return len(self.__gen_blocks)

    @property
    def curr_layer(self) -> int:
        return self.__start_layer
