import torch as th
import torch.nn as nn

from typing import Tuple, List


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
        nb_vec = y.size()[1]

        # (Nb, 2 * Nc, Nvec, 1)
        style = self.__style(y) \
            .view(batch_size, -1, nb_vec, 1)

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

        self.__weights = nn.Parameter(
            th.zeros(1, channels, 1, 1)
        )

    def forward(
            self,
            x: th.Tensor
    ) -> th.Tensor:
        noise = th.randn(
            x.size()[0], 1, x.size()[2], x.size()[3],
            device=x.device
        )

        out = x + self.__weights * noise

        return out

    def __repr__(self):
        return f"NoiseLayer({self.__weights.size()[1]})"

    def __str__(self):
        return self.__repr__()


class BlockEnd(nn.Module):
    def __init__(
            self,
            out_channels: int,
            style_channels: int
    ):
        super(BlockEnd, self).__init__()

        self.__noise = NoiseLayer(out_channels)

        self.__adain = AdaptiveInstanceNorm(
            out_channels, style_channels
        )

    def forward(
            self,
            x: th.Tensor,
            style: th.Tensor
    ) -> th.Tensor:
        out = self.__noise(x)

        out = self.__adain(out, style)

        return out


class InputBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            in_sizes: Tuple[int, int],
            style_channels: int
    ):
        super(InputBlock, self).__init__()

        self.__input = ConstantLayer(
            in_channels,
            in_sizes
        )

        self.__end = BlockEnd(
            in_channels,
            style_channels
        )

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

        out = self.__end(out, style)

        return out


class Block(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int
    ):
        super(Block, self).__init__()

        self.__conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            stride=(2, 2),
            kernel_size=(3, 3),
            padding=(1, 1),
            output_padding=(1, 1)
        )

        self.__end = BlockEnd(
            out_channels,
            style_channels
        )

    def forward(
            self,
            x: th.Tensor,
            style: th.Tensor,
    ) -> th.Tensor:
        out = self.__conv(x)

        out = self.__end(out, style)

        return out


class StyleSmoother(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            out_channels: int,
            num_layers: int = 4
    ):
        super(StyleSmoother, self).__init__()

        self.__gru = nn.GRU(
            input_size=rand_channels,
            hidden_size=out_channels,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, z: th.Tensor) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        nb_batch = z.size()[0]

        h_0 = th.randn(
            self.__gru.num_layers,
            nb_batch,
            self.__gru.input_size,
            device=device
        )

        out, _ = self.__gru(z, h_0)

        return out


class Generator(nn.Module):
    def __init__(
            self,
            style_channels: int
    ):
        super(Generator, self).__init__()

        channels = [
            (256, 256),
            (256, 224),
            (224, 192),
            (192, 160),
            (160, 128),
            (128, 96),
            (96, 64),
            (64, 32)
        ]

        self.__sizes = (2, 2)

        self.__input = InputBlock(
            channels[0][0],
            self.__sizes,
            style_channels
        )

        # Generator layers
        self.__gen_blocks = nn.ModuleList([
            Block(
                c[0], c[1],
                style_channels
            )
            for i, c in enumerate(channels[1:])
        ])

        self.__output = nn.Sequential(
            nn.ConvTranspose2d(
                channels[-1][1],
                2,
                stride=(2, 2),
                kernel_size=(3, 3),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            nn.Tanh()
        )

        self.__style_smoother = StyleSmoother(
            style_channels,
            style_channels,
            num_layers=4
        )

        self.__style_channels = style_channels

    def forward(
            self,
            nb_batch: int,
            nb_input: int = 1
    ) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        z = th.randn(
            nb_batch,
            nb_input * self.__sizes[0] * 1,
            self.__style_channels,
            device=device
        )

        style = self.__style_smoother(z)

        out = self.__input(style, nb_input)

        for i, m in enumerate(self.__gen_blocks):
            z = th.randn(
                nb_batch,
                # + 1 -> count from 1
                # + 1 -> at layer end
                nb_input * self.__sizes[0] ** (i + 2),
                self.__style_channels,
                device=device
            )

            style = self.__style_smoother(z)

            out = m(out, style)

        out = self.__output(out)

        return out

    @property
    def nb_layer(self):
        return self.__gen_blocks.__len__() - 1
