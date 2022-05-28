from typing import Iterator, OrderedDict

import torch as th
import torch.autograd as th_autograd
import torch.nn as nn
import torch.nn.functional as F

from .constants import LEAKY_RELU_SLOPE
from .functions import matrix_multiple
from .layers import FromMagnPhase, PixelNorm


class Block(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(Block, self).__init__(*[
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.LeakyReLU(2e-1),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            nn.LeakyReLU(2e-1),
        ])


class DecBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(DecBlock, self).__init__()

        self.__conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.__conv_down = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1)
        )

        self.__in_channels = in_channels
        self.__out_channels = out_channels

    def forward(self, x: th.Tensor, alpha: float = LEAKY_RELU_SLOPE) -> th.Tensor:
        out = self.__conv(x)
        out = F.leaky_relu(out, alpha)

        out = self.__conv_down(out)
        out = F.leaky_relu(out, alpha)

        return out

    def from_layer(self, factor_2: th.Tensor, bias: th.Tensor) -> None:
        # Init first conv - from last layer
        self.__conv.bias.data[:] = bias.clone()
        nn.init.zeros_(self.__conv.weight)

        self.__conv.weight.data[:, :, 1, 1] = factor_2.transpose(1, 0).clone()

        # Init second conv - identity
        nn.init.zeros_(self.__conv_down.bias)
        nn.init.zeros_(self.__conv_down.weight)

        # with stride of 2, only fill with identity 2 * 2 kernel pixel
        self.__conv_down.weight.data[:, :, 1:, 1:] = (
            th.eye(self.__out_channels)[:, :, None, None]
            .repeat(1, 1, 2, 2) / 4  # kernel is 3 * 3, and we want to fill 2 * 2
        )


class Discriminator(nn.Module):
    def __init__(
            self,
            start_layer: int = 7
    ):
        super(Discriminator, self).__init__()

        conv_channels = [
            (8, 16),
            (16, 24),
            (24, 32),
            (32, 40),
            (40, 48),
            (48, 56),
            (56, 64),
            (64, 72),  # we start here
            (72, 80)
        ]

        self.__grew_up = False

        self.__channels = conv_channels

        self.__curr_layer = start_layer

        stride = 2

        self.__nb_layer = len(conv_channels)
        assert 0 <= start_layer <= len(conv_channels)

        self.__conv_blocks = nn.ModuleList([
            DecBlock(c[0], c[1])
            for i, c in enumerate(conv_channels)
        ])

        self.__start_block = FromMagnPhase(
            conv_channels[start_layer][0]
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
            nn.Linear(out_size, 1),
        )

    def forward(self, x: th.Tensor, alpha: float) -> th.Tensor:
        out = self.__start_block(x, alpha)
        out = self.__conv_blocks[self.__curr_layer](out, alpha)

        for i in range(self.__curr_layer + 1, len(self.__conv_blocks)):
            out = self.__conv_blocks[i](out)

        out = out.flatten(1, -1)

        out_clf = self.__clf(out)

        return out_clf

    def next_layer(self) -> bool:
        if self.growing:
            self.__curr_layer -= 1

            self.__grew_up = True

            last_start_block = self.__start_block

            self.__start_block = FromMagnPhase(
                self.__channels[self.curr_layer][0]
            )

            device = "cuda" \
                if next(self.__conv_blocks.parameters()).is_cuda \
                else "cpu"

            self.__start_block.to(device)

            b = last_start_block.conv.bias.data
            m = last_start_block.conv.weight.data[:, :, 0, 0].transpose(1, 0)
            factor_1, factor_2 = matrix_multiple(m, self.__channels[self.curr_layer][0])

            self.__start_block.from_layer(factor_1)
            self.__conv_blocks[self.__curr_layer].from_layer(factor_2, b)

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

        gradients = gradients[0].view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1.) ** 2.).mean()

        grad_pen_factor = 8.

        return grad_pen_factor * gradient_penalty

    def start_block_parameters(
            self, recurse: bool = True
    ) -> Iterator[nn.Parameter]:
        return self.__start_block.parameters(recurse)

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
        return self.__start_block

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
            batch_first=True
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
