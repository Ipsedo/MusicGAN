import torch as th
import torch.nn as nn

from typing import Iterator


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


class GenActivation(nn.Module):
    def __init__(self, channels: int, style_channels: int):
        super(GenActivation, self).__init__()

        self.__pn = PixelNorm()

        self.__noise = NoiseLayer(
            channels
        )

        self.__adain = AdaIN(
            channels,
            style_channels
        )

        self.__lrelu = nn.LeakyReLU(2e-1)

    def forward(self, x: th.Tensor, style: th.Tensor) -> th.Tensor:
        out = self.__pn(x)
        out = self.__noise(out)
        out = self.__adain(out, style)
        out = self.__lrelu(out)
        return out


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            style_channels: int
    ):
        super(Block, self).__init__()

        self.__conv_1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.__act_1 = GenActivation(
            hidden_channels,
            style_channels
        )

        self.__up_sample = nn.Upsample(
            scale_factor=2.,
            mode="bilinear",
            align_corners=True
        )

        self.__conv_2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.__act_2 = GenActivation(
            out_channels,
            style_channels
        )

    def forward(
            self,
            x: th.Tensor,
            style: th.Tensor
    ) -> th.Tensor:
        out = self.__conv_1(x)
        out = self.__act_1(out, style)

        out = self.__up_sample(out)

        out = self.__conv_2(out)
        out = self.__act_2(out, style)

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
            end_layer: int = 0
    ):
        super(Generator, self).__init__()

        self.__curr_layer = end_layer

        self.__nb_downsample = 8

        channels = [
            (rand_channels, 128, 128),
            (128, 112, 112),
            (112, 96, 96),
            (96, 80, 80),
            (80, 64, 64),
            (64, 48, 48),
            (48, 32, 32),
            (32, 16, 16)
        ]

        self.__channels = channels

        assert 0 <= end_layer < len(channels)

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            Block(
                c[0], c[1], c[2],
                style_channels
            )
            for i, c in enumerate(channels)
        ])

        # for progressive gan
        self.__end_block = ToMagnPhaseLayer(
            channels[self.curr_layer][2]
        )

        self.__last_end_block = (
            None if self.__curr_layer == 0
            else nn.Sequential(
                ToMagnPhaseLayer(
                    channels[self.curr_layer - 1][1]
                ),
                nn.Upsample(
                    scale_factor=2.,
                    mode="bilinear",
                    align_corners=True
                )
            )
        )

        self.__style_network = nn.Sequential(*[
            LinearBlock(style_channels, style_channels)
            for _ in range(8)
        ])

    def forward(
            self,
            z: th.Tensor,
            z_style: th.Tensor,
            alpha: float
    ) -> th.Tensor:
        style = self.__style_network(z_style)

        out = z

        for i in range(self.curr_layer):
            m = self.__gen_blocks[i]
            out = m(out, style)

        m = self.__gen_blocks[self.curr_layer]
        out_last = m(out, style)

        out_mp = self.__end_block(out_last)

        if self.__last_end_block is not None:
            out_old_mp = self.__last_end_block(out)
            return alpha * out_mp + (1. - alpha) * out_old_mp

        return out_mp

    def next_layer(self) -> bool:
        if self.growing:
            self.__curr_layer += 1

            self.__last_end_block = nn.Sequential(
                self.__end_block,
                nn.Upsample(
                    scale_factor=2.,
                    mode="bilinear",
                    align_corners=True
                )
            )

            self.__end_block = ToMagnPhaseLayer(
                self.__channels[self.curr_layer][2]
            )

            device = "cuda" \
                if next(self.__gen_blocks.parameters()).is_cuda \
                else "cpu"

            self.__end_block.to(device)

            return True

        return False

    @property
    def down_sample(self) -> int:
        return self.__nb_downsample

    @property
    def curr_layer(self) -> int:
        return self.__curr_layer

    @property
    def growing(self) -> bool:
        return self.curr_layer < len(self.__gen_blocks) - 1

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return iter(
            list(self.__gen_blocks.parameters(recurse)) +
            list(self.__end_block.parameters(recurse)) +
            list(self.__style_network.parameters(recurse))
        )
