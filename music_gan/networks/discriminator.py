import torch as th
import torch.nn as nn
import torch.autograd as th_autograd

from .layers import FromMagnPhase

from typing import Iterator


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
            (64, 72),
            (72, 80)
        ]

        self.__channels = conv_channels

        self.__curr_layer = start_layer

        stride = 2

        self.__nb_layer = len(conv_channels)
        assert 0 <= start_layer <= len(conv_channels)

        self.__conv_blocks = nn.ModuleList([
            Block(c[0], c[1])
            for i, c in enumerate(conv_channels)
        ])

        self.__last_start_block = None

        self.__start_block = FromMagnPhase(
            conv_channels[self.curr_layer][0]
        )

        nb_time = 512
        nb_freq = 512

        out_size = (
                conv_channels[-1][1] *
                nb_time // stride ** self.__nb_layer *
                nb_freq // stride ** self.__nb_layer
        )

        self.__clf = nn.Sequential(
            nn.Linear(out_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor, alpha: float) -> th.Tensor:
        out_new = self.__start_block(x)
        out_new = self.__conv_blocks[self.__curr_layer](out_new)

        if self.__last_start_block:
            out_old = self.__last_start_block(x)
            out = alpha * out_new + (1 - alpha) * out_old
        else:
            out = out_new

        for i in range(self.__curr_layer + 1, len(self.__conv_blocks)):
            out = self.__conv_blocks[i](out)

        out = out.flatten(1, -1)

        out_clf = self.__clf(out)

        return out_clf

    def next_layer(self) -> bool:
        if self.growing:
            self.__curr_layer -= 1

            self.__last_start_block = nn.Sequential(
                nn.AvgPool2d(2, 2),
                self.__start_block
            )

            self.__start_block = FromMagnPhase(
                self.__channels[self.curr_layer][0]
            )

            device = "cuda" \
                if next(self.__conv_blocks.parameters()).is_cuda \
                else "cpu"

            self.__start_block.to(device)

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

        out_interpolated = self(x_interpolated, alpha)

        gradients = th_autograd.grad(
            out_interpolated, x_interpolated,
            grad_outputs=th.ones(out_interpolated.size(), device=device),
            create_graph=True, retain_graph=True
        )

        gradients = gradients[0].view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1.) ** 2.).mean()

        grad_pen_factor = 20.

        return grad_pen_factor * gradient_penalty

    def start_block_parameters(self) -> Iterator[nn.Parameter]:
        return self.__start_block.parameters()
