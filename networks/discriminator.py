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
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
            nn.LeakyReLU(2e-1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__conv(x)


class Discriminator(nn.Module):
    def __init__(
            self,
            in_channels: int
    ):
        super(Discriminator, self).__init__()

        conv_channels = [
            (in_channels, 32),
            (32, 64),
            (64, 96),
            (96, 128),
            (128, 160),
            (160, 192),
            (192, 224),
            (224, 256),
            (256, 256)
        ]

        stride = 2

        nb_layer = 9

        self.__conv = nn.Sequential(*[
            ConvBlock(
                conv_channels[i][0],
                conv_channels[i][1]
            )
            for i in range(nb_layer)
        ])

        nb_time = 512
        nb_freq = 512

        out_size = conv_channels[-1][1] * \
                   nb_time // stride ** nb_layer * \
                   nb_freq // stride ** nb_layer

        self.__clf = nn.Linear(out_size, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv = self.__conv(x)
        out = out_conv.flatten(1, -1)

        out_clf = self.__clf(out)

        return out_clf

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
