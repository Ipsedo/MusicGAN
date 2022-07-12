from typing import Iterator, OrderedDict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .constants import LEAKY_RELU_SLOPE
from .functions import matrix_multiple
from .layers import PixelNorm, ToMagnPhase, LayerNorm2d, Conv2dPadding


class GenBlock(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(GenBlock, self).__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1)
            ),
            nn.SELU(),

            nn.Upsample(
                scale_factor=2.,
                mode="nearest"
            ),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1)
            ),
            nn.SELU(),
        )


class LinearityFadeinBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(LinearityFadeinBlock, self).__init__()

        self.__conv_1 = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
        )

        self.__up = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1)
        )

        self.__conv_2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1)
        )

        self.__in_channels = in_channels
        self.__out_channels = out_channels

    def forward(self, x: th.Tensor, slope: float = LEAKY_RELU_SLOPE) -> th.Tensor:
        out = self.__conv_1(x)
        out = F.leaky_relu(out, slope)

        out = self.__up(out)
        out = F.leaky_relu(out, slope)

        out = self.__conv_2(out)
        out = F.leaky_relu(out, slope)

        return out

    def from_layer(self, factor_1: th.Tensor) -> None:
        # Init first conv - identity
        nn.init.zeros_(self.__conv_1.bias)
        nn.init.zeros_(self.__conv_1.weight)

        self.__conv_1.weight.data[:, :, 1, 1] = (
            th.eye(self.__in_channels)
        )

        # Init strided transposed conv
        nn.init.zeros_(self.__up.bias)
        nn.init.zeros_(self.__up.weight)
        self.__up.weight.data[:, :, 1:, 1:] = (
            th.eye(self.__in_channels)[:, :, None, None]
            .repeat(1, 1, 2, 2)
        )

        # Init second conv - from last layer
        nn.init.zeros_(self.__conv_2.bias)
        nn.init.zeros_(self.__conv_2.weight)

        self.__conv_2.weight.data[:, :, 1, 1] = factor_1.clone()


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
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

        assert 0 <= end_layer < len(channels), \
            f"0 <= {end_layer} < {len(channels)}"

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            LinearityFadeinBlock(c[0], c[1])
            for c in channels
        ])

        # for progressive gan
        self.__end_blocks = nn.ModuleList(
            ToMagnPhase(c[1])
            for c in channels
        )

    def forward(
            self,
            z: th.Tensor,
            slope: float
    ) -> th.Tensor:
        out = z

        for i in range(self.curr_layer):
            out = self.__gen_blocks[i](out)

        out_block = self.__gen_blocks[self.curr_layer](out, slope)
        out_mp = self.__end_blocks[self.curr_layer](out_block)

        return out_mp

    def next_layer(self) -> bool:
        if self.growing:
            self.__curr_layer += 1

            b = self.__end_blocks[self.curr_layer - 1].conv.bias.data
            m = self.__end_blocks[self.curr_layer - 1].conv.weight.data[:, :, 0, 0].transpose(1, 0)
            factor_1, factor_2 = matrix_multiple(m, self.__channels[self.curr_layer][1])

            self.__gen_blocks[self.curr_layer].from_layer(factor_1.transpose(1, 0))
            self.__end_blocks[self.curr_layer].from_layer(factor_2.transpose(1, 0), b)

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

    @property
    def conv_blocks(self) -> nn.ModuleList:
        return self.__gen_blocks

    @property
    def end_block(self) -> nn.Module:
        return self.__end_blocks

    def end_block_parameters(
            self, recurse: bool = True
    ) -> Iterator[nn.Parameter]:
        return self.__end_blocks.parameters(recurse)


############
# Recurrent
############

class RecurrentGenerator(nn.Module):
    def __init__(
            self,
            input_size:int,
            conv_rand_channels: int,
            cnn_state_dict: OrderedDict[str, th.Tensor]
    ):
        super(RecurrentGenerator, self).__init__()

        self.__rnn = nn.RNN(
            input_size,
            conv_rand_channels * 2,
            batch_first=True,
            # We want [-1; 1] to approximately fit N(0; 1)
            nonlinearity="tanh"
        )

        gen = Generator(
            conv_rand_channels, end_layer=7
        )

        gen.load_state_dict(cnn_state_dict)

        self.__conv_blocks = gen.conv_blocks
        self.__end_block = gen.end_block

    def forward(self, z_rec: th.Tensor) -> th.Tensor:

        out_rec, _ = self.__rnn(z_rec)

        out = (
            # split on freq dim
            th.stack(out_rec.split(16, dim=-1), dim=1)
            # batch, channels, freq, time
            .permute(0, 3, 1, 2)
        )

        for layer in self.__conv_blocks:
            out = layer(out)

        out = self.__end_block(out)

        return out
