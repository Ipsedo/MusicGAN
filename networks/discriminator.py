import torch as th
import torch.nn as nn
import torch.autograd as th_autograd


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(ConvBlock, self).__init__()

        self.__conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.AvgPool2d(2, 2),
            nn.LeakyReLU(2e-1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__conv(x)


class MagPhaseLayer(nn.Module):
    def __init__(self, out_channels: int):
        super(MagPhaseLayer, self).__init__()

        self.__conv = nn.Sequential(
            nn.Conv2d(
                2, out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.LeakyReLU(2e-1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__conv(x)


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            start_layer: int = 7
    ):
        super(Discriminator, self).__init__()

        assert 0 <= start_layer <= 7

        conv_channels = [
            (in_channels, 32),  # 512
            (32, 64),  # 256
            (64, 96),  # 128
            (96, 128),  # 64
            (128, 160),  # 32
            (160, 192),  # 16
            (192, 224),  # 8
            (224, 256),  # 4
            (256, 288)  # 2
            # 1
        ]

        # +1 for gen end_block
        self.__curr_layer = start_layer

        stride = 2

        nb_layer = 9

        self.__conv_blocks = nn.ModuleList([
            ConvBlock(
                conv_channels[i][0],
                conv_channels[i][1]
            )
            for i in range(nb_layer)
        ])

        self.___start_blocks = nn.ModuleList([
            MagPhaseLayer(c[1])
            for c in conv_channels[:-1]
        ])

        nb_time = 512
        nb_freq = 512

        out_size = conv_channels[-1][1] * \
                   nb_time // stride ** nb_layer * \
                   nb_freq // stride ** nb_layer

        self.__clf = nn.Linear(out_size, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:

        out = x

        if self.__curr_layer >= 1:
            # -1 disc has one layer more then gen
            out = self.___start_blocks[self.__curr_layer - 1](out)

        for i in range(self.__curr_layer, len(self.__conv_blocks)):
            out = self.__conv_blocks[i](out)

        out = out.flatten(1, -1)

        out_clf = self.__clf(out)

        return out_clf

    def next_layer(self) -> None:
        self.__curr_layer -= 1

        if self.__curr_layer < 1:
            self.__curr_layer = 0

    def gradient_penalty(
            self,
            x_real: th.Tensor,
            x_gen: th.Tensor
    ) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        batch_size = x_real.size()[0]
        eps = th.rand(batch_size, 1, 1, 1, device=device)

        x_interpolated = eps * x_real + (1 - eps) * x_gen

        out_interpolated = self(x_interpolated)

        gradients = th_autograd.grad(
            out_interpolated, x_interpolated,
            grad_outputs=th.ones(out_interpolated.size(), device=device),
            create_graph=True, retain_graph=True
        )

        gradients = gradients[0].view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1.) ** 2.).mean()

        grad_pen_factor = 10.

        return grad_pen_factor * gradient_penalty
