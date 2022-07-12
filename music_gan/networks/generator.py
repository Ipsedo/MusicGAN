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
            nn.LeakyReLU(LEAKY_RELU_SLOPE),

            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1)
            ),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
        )


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            end_layer: int = 0
    ):
        super(Generator, self).__init__()

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
        self.__gen_blocks = nn.Sequential(*[
            GenBlock(c[0], c[1])
            for c in channels
        ])

        self.__end_block = ToMagnPhase(channels[-1][1])

    def forward(
            self,
            z: th.Tensor
    ) -> th.Tensor:
        out = self.__gen_blocks(z)
        out_mp = self.__end_block(out)

        return out_mp

    @property
    def down_sample(self) -> int:
        return self.__nb_downsample

    @property
    def conv_blocks(self) -> nn.ModuleList:
        return self.__gen_blocks

    @property
    def end_block(self) -> nn.Module:
        return self.__end_blocks


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
