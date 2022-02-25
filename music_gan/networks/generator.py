import torch as th
import torch.nn as nn

from typing import Iterator

from .layers import AdaIN, NoiseLayer, PixelNorm


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int
    ):
        super(ConvBlock, self).__init__()

        self.__conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.__noise = NoiseLayer(
            out_channels
        )

        self.__adain = AdaIN(
            out_channels,
            style_channels
        )

        self.__lr = nn.LeakyReLU(
            2e-1
        )

        self.__pn = PixelNorm()

    def forward(self, x: th.Tensor, style: th.Tensor) -> th.Tensor:
        out = self.__conv(x)
        out = self.__noise(out)
        out = self.__adain(out, style)
        out = self.__lr(out)
        out = self.__pn(out)

        return out

    def __repr__(self) -> str:
        return "ConvBlock(\n" +\
               f' 0: {self.__conv},\n' + \
               f" 1: {self.__noise},\n" + \
               f" 2: {self.__adain},\n" + \
               f" 3: {self.__lr},\n" + \
               f" 4: {self.__pn}\n" + \
               ")"

    def __str__(self) -> str:
        return self.__repr__()


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int
    ):
        super(Block, self).__init__()

        self.__block_1 = ConvBlock(
            in_channels,
            in_channels,
            style_channels
        )

        self.__up_sample = nn.Upsample(
            scale_factor=2.,
            mode="nearest"
        )

        self.__block_2 = ConvBlock(
            in_channels,
            out_channels,
            style_channels
        )

    def forward(self, x: th.Tensor, style: th.Tensor) -> th.Tensor:
        out = self.__block_1(x, style)
        out = self.__up_sample(out)
        out = self.__block_2(out, style)

        return out

    def __repr__(self) -> str:
        return "Block(\n" + \
               f" 0: {self.__block_1},\n" + \
               f" 1: {self.__up_sample},\n" + \
               f" 2: {self.__block_2}\n" + \
               ")"

    def __str__(self) -> str:
        return self.__repr__()


class ToMagnPhaseLayer(nn.Sequential):
    def __init__(self, in_channels: int):
        super(ToMagnPhaseLayer, self).__init__(
            nn.Conv2d(
                in_channels, 2,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            nn.Tanh()
        )


class LinearBlock(nn.Sequential):
    def __init__(self, in_size: int, out_size: int):
        super(LinearBlock, self).__init__(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(2e-1)
        )


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            rand_style_size: int,
            end_layer: int = 0
    ):
        super(Generator, self).__init__()

        self.__curr_layer = end_layer

        self.__nb_downsample = 7

        channels = [
            (rand_channels, 64),
            (64, 56),
            (56, 48),
            (48, 40),
            (40, 32),
            (32, 24),
            (24, 16),
            (16, 8)
        ]

        self.__channels = channels
        self.__style_channels = 32

        assert 0 <= end_layer < len(channels), \
            f"0 <= {end_layer} < {len(channels)}"

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            Block(c[0], c[1], self.__style_channels)
            for i, c in enumerate(channels)
        ])

        # for progressive gan
        self.__end_block = ToMagnPhaseLayer(
            channels[self.curr_layer][1]
        )

        self.__last_end_block = (
            None if self.__curr_layer == 0
            else nn.Sequential(
                nn.Upsample(
                    scale_factor=2.,
                    mode="nearest"
                ),
                ToMagnPhaseLayer(
                    channels[self.curr_layer - 1][1]
                )
            )
        )

        style_sizes = [
            (rand_style_size, self.__style_channels),
            *[
                (self.__style_channels, self.__style_channels)
                for _ in range(7)
            ]
        ]

        self.__style_network = nn.Sequential(*[
            LinearBlock(
                in_size, out_size
            )
            for in_size, out_size in style_sizes
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

        out_block = self.__gen_blocks[self.curr_layer](out, style)

        out_mp = self.__end_block(out_block)

        if self.__last_end_block is not None:
            out_old_mp = self.__last_end_block(out)
            return alpha * out_mp + (1. - alpha) * out_old_mp

        return out_mp

    def next_layer(self) -> bool:
        if self.growing:
            self.__curr_layer += 1

            self.__last_end_block = nn.Sequential(
                nn.Upsample(
                    scale_factor=2.,
                    mode="nearest"
                ),
                self.__end_block
            )

            self.__end_block = ToMagnPhaseLayer(
                self.__channels[self.curr_layer][1]
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

    def end_block_params(self) -> Iterator[nn.Parameter]:
        return self.__end_block.parameters()

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self.parameters():
            p.grad = None

    # def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
    #     return iter(
    #         list(self.__end_block.parameters(recurse)) +
    #         list(self.__gen_blocks.parameters(recurse)) +
    #         list(self.__style_network.parameters(recurse))
    #     )
