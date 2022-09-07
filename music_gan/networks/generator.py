from typing import Iterator, OrderedDict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .constants import LEAKY_RELU_SLOPE
from .layers import PixelNorm, ToMagnPhase, EqualLrConvTr2d, EqualLrConv2d


class GenBlock(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(GenBlock, self).__init__(
            EqualLrConv2d(
                in_channels,
                in_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                alpha=1.
            ),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            PixelNorm(),

            nn.Upsample(
                scale_factor=2.,
                mode="nearest"
            ),

            EqualLrConv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                alpha=1.
            ),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            PixelNorm(),
        )


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            end_layer: int = 0
    ):
        super(Generator, self).__init__()

        self.__grew_up = False

        self.__curr_layer = end_layer

        self.__nb_downsample = 7

        channels = [
            (rand_channels, 512),
            (512, 512),
            (512, 512),
            (512, 256),
            (256, 128),
            (128, 64),
            (64, 32),
            (32, 16)
        ]

        self.__channels = channels

        assert 0 <= end_layer < len(channels), \
            f"0 <= {end_layer} < {len(channels)}"

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            GenBlock(c[0], c[1])
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
            alpha: float
    ) -> th.Tensor:
        out = z

        for i in range(self.curr_layer):
            out = self.__gen_blocks[i](out)

        out_block = self.__gen_blocks[self.curr_layer](out)
        out_mp = self.__end_blocks[self.curr_layer](out_block)

        if self.__grew_up:
            out_old = F.interpolate(
                self.__end_blocks[self.curr_layer - 1](out),
                scale_factor=2.,
                mode="nearest"
            )

            out_mp = out_old * (1. - alpha) + out_mp * alpha

        return out_mp

    def next_layer(self) -> bool:
        if self.growing:
            self.__curr_layer += 1

            self.__grew_up = True

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
