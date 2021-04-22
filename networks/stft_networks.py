import torch as th
import torch.nn as nn


# Generator

class ResidualTransConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            kernel_size: int,
            kernel_size_up_sample: int,
            up_sample: int
    ):
        assert kernel_size % 2 == 1
        assert kernel_size_up_sample % 2 == 0
        assert up_sample % 2 == 0

        super(ResidualTransConv, self).__init__()

        self.__tr_conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels, hidden_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels, in_channels,
                kernel_size,
                stride=1,
                padding=kernel_size // 2

            ),
            nn.ReLU()
        )

        self.__upsample_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size_up_sample,
                stride=up_sample,
                padding=kernel_size // 2
            ),
            nn.ReLU()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__tr_conv_block(x)

        out = x + out
        out = self.__upsample_conv(out)

        return out


# Gate Unit
class GatedActUnit(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            output_channels: int,
            kernel_size: int,
            stride: int
    ):
        assert kernel_size % 2 == 0

        super(GatedActUnit, self).__init__()

        self.__filter_conv = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=kernel_size + 1,
            stride=1,
            padding=kernel_size // 2
        )

        self.__gate_conv = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=kernel_size + 1,
            stride=1,
            padding=kernel_size // 2
        )

        self.__tr_conv = nn.ConvTranspose2d(
            hidden_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - stride) // 2
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_f = self.__filter_conv(x)
        out_g = self.__gate_conv(x)

        out = th.tanh(out_f) * th.sigmoid(out_g)

        out = th.relu(self.__tr_conv(out))

        return out


class STFTGenerator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            hidden_channel: int,
            residual_channel: int,
            out_channel: int
    ):
        super(STFTGenerator, self).__init__()

        nb_layer = 7

        """self.__gen = nn.Sequential(*[
            ResidualTransConv(
                rand_channels if i == 0 else residual_channel,
                hidden_channel,
                residual_channel,
                5, 6, 2
            )
            for i in range(nb_layer)
        ])"""

        self.__gen = nn.Sequential(*[
            GatedActUnit(
                rand_channels if i == 0 else residual_channel,
                hidden_channel,
                residual_channel,
                4, 2
            )
            for i in range(nb_layer)
        ])

        self.__conv_out = nn.Sequential(
            nn.Conv2d(
                residual_channel,
                out_channel,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.Tanh()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__gen(x)

        out = self.__conv_out(out)

        return out


# Discriminator (designed for "image" of 2 * 256 * 512 shape)

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int
    ):
        super(ConvBlock, self).__init__()

        self.__conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__conv(x)

        return th.relu(out)


class STFTDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channel: int
    ):
        super(STFTDiscriminator, self).__init__()

        nb_layer = 5

        self.__conv = nn.Sequential(*[
            ConvBlock(
                in_channels if i == 0 else hidden_channel,
                hidden_channel,
                kernel_size=5,
                stride=2
            )
            for i in range(nb_layer)
        ])

        nb_time = 256
        nb_freq = 512

        out_size = hidden_channel * nb_time // 2 ** nb_layer * nb_freq // 2 ** nb_layer

        self.__clf = nn.Sequential(
            nn.Linear(out_size, 2560),
            nn.ReLU(),
            nn.Linear(2560, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv = self.__conv(x)
        out = out_conv.flatten(1, -1)

        out_clf = self.__clf(out)

        return out_clf


if __name__ == '__main__':
    rand_data = th.rand(1, 8, 2, 4)

    # rs = ResidualTransConv(8, 24, 16, 3, 4, 2)
    gu = GatedActUnit(
        8, 10, 6, 2
    )

    o = gu(rand_data)

    print(o.size())

    gen = STFTGenerator(
        8, 24, 32, 2
    )

    print(gen)

    o = gen(rand_data)

    print("A ", o.size())

    disc = STFTDiscriminator(
        2, 64
    )

    print(disc)

    o = disc(o)

    print(o.size())
