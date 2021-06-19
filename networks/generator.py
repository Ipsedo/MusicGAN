import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, List

from .utils import tile


class ConstantLayer(nn.Module):
    def __init__(
            self,
            channels: int,
            sizes: Tuple[int, int]
    ):
        super(ConstantLayer, self).__init__()

        self.__constant = nn.Parameter(
            th.randn(1, channels, sizes[0], sizes[1])
        )

    def forward(self, batch_size: int) -> th.Tensor:
        constant = self.__constant.repeat(batch_size, 1, 1, 1)

        return constant

    def __repr__(self):
        return f"ConstantLayer(" \
               f"{self.__constant.size()[1]}, " \
               f"size=({self.__constant.size()[2]}, " \
               f"{self.__constant.size()[3]}))"

    def __str__(self):
        return self.__repr__()


class AdaptiveInstanceNorm(nn.Module):
    def __init__(
            self,
            in_channels: int,
            style_channels: int
    ):
        super(AdaptiveInstanceNorm, self).__init__()

        self.__style = nn.Linear(style_channels, in_channels * 2)

    def forward(
            self,
            x: th.Tensor,
            y: th.Tensor
    ) -> th.Tensor:
        """

        :param x:
        :type x:
        :param y: style, size=(Nb, Nvec, Nc)
        :type y:
        :return:
        :rtype:
        """
        batch_size = x.size()[0]
        in_channels = x.size()[1]

        # (Nb, 2 * Nc, 1, 1)
        style = self.__style(y) \
            .view(batch_size, -1, 1, 1)

        mean = x.view(batch_size, in_channels, -1) \
            .mean(dim=2) \
            .view(batch_size, in_channels, 1, 1)

        std = x.view(batch_size, in_channels, -1) \
            .std(dim=2) \
            .view(batch_size, in_channels, 1, 1)

        y_b, y_s = style.chunk(2, 1)

        x_norm = (x - mean) / std

        out = x_norm * y_s + y_b

        return out

    def __repr__(self):
        return f"AdaptiveInstanceNorm(" \
               f"style_channels={self.__style.weight.size()[1]}, " \
               f"in_channels={self.__style.weight.size()[0] // 2})"

    def __str__(self):
        return self.__repr__()


class NoiseLayer(nn.Module):
    def __init__(
            self,
            channels: int
    ):
        super(NoiseLayer, self).__init__()

        self.__channels = channels

        self.__to_noise = nn.Linear(1, channels)

    def forward(
            self,
            x: th.Tensor
    ) -> th.Tensor:
        noise = th.randn(
            x.size()[0], x.size()[2], x.size()[3], 1,
            device=x.device
        )

        out = x + self.__to_noise(noise).permute(0, 3, 1, 2)

        return out

    def __repr__(self):
        return f"NoiseLayer({self.__channels})"


class Conv2DMod(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int,
            epsilon: float = 1e-8
    ):
        super(Conv2DMod, self).__init__()

        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__style_channels = style_channels

        self.__kernel_size = 3
        self.__stride = 1
        self.__dilation = 1

        self.__weights = nn.Parameter(th.randn(
            out_channels, in_channels,
            self.__kernel_size,
            self.__kernel_size
        ))

        self.__biais = nn.Parameter(th.zeros(
            out_channels
        ))

        self.__style = nn.Linear(style_channels, in_channels)

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        batch_size = x.size()[0]
        width, height = x.size()[2], x.size()[3]
        in_c, out_c = self.__weights.size()[1], \
                      self.__weights.size()[0]

        # y = (nb, style)
        style = self.__style(y)

        # weight = (out, in, ker, ker)
        # style = (nb, in)

        # (1, out, in, ker, ker)
        weights = self.__weights.unsqueeze(0)
        # (nb, 1, in, 1, 1)
        style = style.view(batch_size, 1, -1, 1, 1)

        w_mod = weights * style

        w_std = th.sqrt(
            w_mod.pow(2.).sum(
                dim=(2, 3, 4),
                keepdim=True
            ) +
            self.__epsilon
        )

        w_demod = w_mod / w_std

        weights = w_demod.view(
            batch_size * out_c,
            in_c,
            self.__kernel_size,
            self.__kernel_size
        )

        x = x.view(1, -1, width, height)

        out = F.conv2d(
            x, weights,
            None,
            padding="same",
            groups=batch_size
        )

        out = out.view(batch_size, out_c, width, height)

        out = out + self.__biais.view(1, -1, 1, 1)

        return out

    def __repr__(self):
        return f"Conv2DStyle(" \
               f"{self.__in_channels}, " \
               f"{self.__out_channels}, " \
               f"kernel_size=(3, 3), " \
               f"stride=(1, 1), " \
               f"padding=(1, 1), " \
               f"style_channels={self.__style_channels}" \
               f")"

    def __str__(self):
        return self.__repr__()


class InputBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            in_sizes: Tuple[int, int],
            style_channels: int
    ):
        super(InputBlock, self).__init__()

        self.__input = ConstantLayer(
            in_channels,
            in_sizes
        )

        self.__conv_style = Conv2DMod(
            in_channels,
            out_channels,
            style_channels
        )

        self.__noise = NoiseLayer(out_channels)

    def forward(
            self,
            style: th.Tensor,
            nb_input: int = 1
    ) -> th.Tensor:
        batch_size = style.size()[0]

        out = th.cat([
            self.__input(batch_size)
            for _ in range(nb_input)
        ], dim=2)

        out = self.__conv_style(out, style)
        out = self.__noise(out)

        return out


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int,
            last_layer: bool
    ):
        super(Block, self).__init__()

        self.__up_sample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.__conv = Conv2DMod(
            in_channels,
            out_channels,
            style_channels
        )

        self.__end = NoiseLayer(out_channels) \
            if not last_layer else nn.Tanh()

    def forward(
            self,
            x: th.Tensor,
            style: th.Tensor,
    ) -> th.Tensor:
        out = self.__up_sample(x)

        out = self.__conv(out, style)

        out = self.__end(out)

        return out


class LinearLayer(nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(LinearLayer, self).__init__(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(2e-1)
        )


class Generator(nn.Module):
    def __init__(
            self,
            style_channels: int
    ):
        super(Generator, self).__init__()

        channels = [
            (256, 224),
            (224, 192),
            (192, 160),
            (160, 128),
            (128, 96),
            (96, 64),
            (64, 32),
            (32, 2)
        ]

        self.__sizes = (2, 2)

        self.__input = InputBlock(
            channels[0][0],
            channels[0][0],
            self.__sizes,
            style_channels
        )

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            Block(
                c[0], c[1],
                style_channels,
                i == len(channels) - 1
            )
            for i, c in enumerate(channels)
        ])

        self.__style_network = nn.Sequential(*[
            LinearLayer(style_channels, style_channels)
            for _ in range(4)
        ])

        self.__style_channels = style_channels

    def forward(
            self,
            nb_batch: int,
            nb_input: int = 1
    ) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        z = th.randn(
            nb_batch,
            self.__style_channels,
            device=device
        )

        style = self.__style_network(z)

        out = self.__input(style, nb_input)

        for m in self.__gen_blocks:
            out = m(out, style)

        return out

    @property
    def nb_layer(self):
        return self.__gen_blocks.__len__() - 1
