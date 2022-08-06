from typing import Iterator, OrderedDict

import torch as th
import torch.autograd as th_autograd
import torch.nn as nn
import torch.nn.functional as F

from .constants import LEAKY_RELU_SLOPE
from .layers import FromMagnPhase, PixelNorm, EqualLrConv2d, EqualLrLinear, MiniBatchStdDev


class DiscBlock(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mini_batch_std_dev: bool = False
    ):
        if mini_batch_std_dev:
            in_channels += 1

        super(DiscBlock, self).__init__(
            MiniBatchStdDev() if mini_batch_std_dev
            else nn.Identity(),

            EqualLrConv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),

            nn.AvgPool2d(2, 2),

            EqualLrConv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
        )


class Discriminator(nn.Module):
    def __init__(
            self,
            start_layer: int = 7
    ):
        super(Discriminator, self).__init__()

        self.__grew_up = False

        conv_channels = [
            (16, 32),
            (32, 48),
            (48, 64),
            (64, 80),
            (80, 96),
            (96, 112),
            (112, 128),
            (128, 144), # we start here
            (144, 160)
        ]

        self.__channels = conv_channels

        self.__curr_layer = start_layer

        stride = 2

        self.__nb_layer = len(conv_channels)
        assert 0 <= start_layer <= len(conv_channels)

        self.__conv_blocks = nn.ModuleList(
            DiscBlock(c[0], c[1], i == len(conv_channels) - 1)
            for i, c in enumerate(conv_channels)
        )

        self.__start_blocks = nn.ModuleList(
            FromMagnPhase(c[0])
            for c in conv_channels[:-1]
        )

        nb_time = 512
        nb_freq = 512

        # for recurrent we won't keep the last block
        self.__end_layer_channels = conv_channels[-2][1]

        out_size = (
                conv_channels[-1][1] *
                nb_time // stride ** self.__nb_layer *
                nb_freq // stride ** self.__nb_layer
        )

        self.__clf = nn.Sequential(
            EqualLrLinear(out_size, 1)
        )

    def forward(self, x: th.Tensor, alpha: float) -> th.Tensor:
        out = self.__start_blocks[self.curr_layer](x)
        out = self.__conv_blocks[self.curr_layer](out)

        if self.__grew_up:
            out_old = self.__start_blocks[self.curr_layer + 1](
                F.avg_pool2d(x, (2, 2))
            )
            out = out_old * (1. - alpha) + out * alpha

        for i in range(self.curr_layer + 1, len(self.__conv_blocks)):
            out = self.__conv_blocks[i](out)

        out = out.flatten(1, -1)

        out_clf = self.__clf(out)

        return out_clf

    def next_layer(self) -> bool:
        if self.growing:
            self.__curr_layer -= 1

            self.__grew_up = True

            return True

        return False

    @property
    def curr_layer(self) -> int:
        return self.__curr_layer

    @property
    def growing(self) -> bool:
        return self.__curr_layer > 0

    def gradient_penalty(
            self,
            x_real: th.Tensor,
            x_gen: th.Tensor,
            alpha: float
    ) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        batch_size = x_real.size()[0]
        eps = th.rand(batch_size, 1, 1, 1, device=device)

        x_interpolated = eps * x_real + (1 - eps) * x_gen
        x_interpolated.requires_grad_(True)

        out_interpolated = self(x_interpolated, alpha)

        gradients = th_autograd.grad(
            out_interpolated, x_interpolated,
            grad_outputs=th.ones(out_interpolated.size(), device=device),
            create_graph=True, retain_graph=True
        )

        grad_objective = 1.
        grad_pen_factor = 8.

        gradients = gradients[0].view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - grad_objective) ** 2.).mean()

        return grad_pen_factor * gradient_penalty

    def start_block_parameters(
            self, recurse: bool = True
    ) -> Iterator[nn.Parameter]:
        raise NotImplementedError()
        #return self.__start_block.parameters(recurse)

    @property
    def conv_blocks(self) -> nn.ModuleList:
        return nn.ModuleList([
            layer
            for i, layer in enumerate(self.__conv_blocks)
            # skip last layer
            if i < len(self.__conv_blocks) - 1
        ])

    @property
    def start_block(self) -> nn.Module:
        return self.__start_blocks[self.curr_layer]

    @property
    def end_layer_channels(self) -> int:
        return self.__end_layer_channels


############
# Recurrent
############

class RecurrentDiscriminator(nn.Module):
    def __init__(self, cnn_state_dict: OrderedDict[str, th.Tensor]):
        super(RecurrentDiscriminator, self).__init__()

        conv_disc = Discriminator(start_layer=0)
        conv_disc.load_state_dict(cnn_state_dict)

        self.__start_block = conv_disc.start_block
        self.__conv_blocks = conv_disc.conv_blocks

        rnn_out_size = 64

        self.__rnn = nn.RNN(
            conv_disc.end_layer_channels * 2,
            rnn_out_size,
            batch_first=True,
            nonlinearity="tanh"  # Use ReLU or tanh ? (symmetry with generator)
        )

        self.__clf = nn.Linear(
            rnn_out_size,
            1
        )

    def forward(self, data: th.Tensor) -> th.Tensor:
        out = self.__start_block(data)
        for layer in self.__conv_blocks:
            out = layer(out)

        out = (
            # flatten channels and freq
            th.flatten(out, 1, 2)
            # permute <0: batch, 2: time, 1: channels * freq>
            .permute(0, 2, 1)
        )

        out, _ = self.__rnn(out)

        out = self.__clf(out)
        out = out.mean(dim=1)

        return out
