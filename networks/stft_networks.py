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
                (kernel_size, kernel_size),
                stride=(1, 1),
                padding=(
                    kernel_size // 2,
                    kernel_size // 2
                )
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels, in_channels,
                (kernel_size, kernel_size),
                stride=(1, 1),
                padding=(
                    kernel_size // 2,
                    kernel_size // 2
                )
            ),
            nn.ReLU()
        )

        self.__upsample_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                (kernel_size_up_sample, kernel_size_up_sample),
                stride=(up_sample, up_sample),
                padding=(
                    kernel_size // 2,
                    kernel_size // 2
                )
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
            conv_ker_size: int,
            tr_conv_ker_size: int,
            stride: int
    ):
        assert conv_ker_size % 2 == 1
        assert tr_conv_ker_size % 2 == 0

        super(GatedActUnit, self).__init__()

        self.__filter_conv = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=(
                conv_ker_size,
                conv_ker_size
            ),
            stride=(1, 1),
            padding=(
                conv_ker_size // 2,
                conv_ker_size // 2
            )
        )

        self.__gate_conv = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=(
                conv_ker_size,
                conv_ker_size
            ),
            stride=(1, 1),
            padding=(
                conv_ker_size // 2,
                conv_ker_size // 2
            )
        )

        self.__tr_conv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels,
                output_channels,
                kernel_size=(
                    tr_conv_ker_size,
                    tr_conv_ker_size
                ),
                stride=(stride, stride),
                padding=(
                    (tr_conv_ker_size - stride) // 2,
                    (tr_conv_ker_size - stride) // 2
                )
            ),
            nn.LeakyReLU(negative_slope=1e-1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_f = self.__filter_conv(x)
        out_g = self.__gate_conv(x)

        out = th.tanh(out_f) * th.sigmoid(out_g)

        out = self.__tr_conv(out)

        return out


class STFTGenerator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            out_channel: int
    ):
        super(STFTGenerator, self).__init__()

        nb_layer = 4

        """self.__gen = nn.Sequential(*[
            ResidualTransConv(
                rand_channels if i == 0 else residual_channel,
                hidden_channel,
                residual_channel,
                5, 6, 2
            )
            for i in range(nb_layer)
        ])"""

        channel_list = [
            224,
            112,
            56,
            28
        ]

        h_channel_list = [
            256,
            128,
            64,
            32
        ]

        self.__gen = nn.Sequential(*[
            GatedActUnit(
                rand_channels if i == 0
                else channel_list[i - 1],
                h_channel_list[i],
                channel_list[i],
                5, 6, 4
            )
            for i in range(nb_layer)
        ])

        self.__conv_out = nn.Sequential(
            nn.Conv2d(
                channel_list[-1],
                out_channel,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=(2, 2)
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
            (kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(
                kernel_size // 2,
                kernel_size // 2
            )
        )

        self.__relu = nn.LeakyReLU(negative_slope=1e-1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__conv(x)

        return self.__relu(out)


class STFTDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int
    ):
        super(STFTDiscriminator, self).__init__()

        nb_layer = 3
        stride = 4

        hidden_channels = [
            16,
            32,
            64
        ]

        self.__conv = nn.Sequential(*[
            ConvBlock(
                in_channels if i == 0 else hidden_channels[i - 1],
                hidden_channels[i],
                kernel_size=5,
                stride=stride
            )
            for i in range(nb_layer)
        ])

        nb_time = 256
        nb_freq = 512

        out_size = hidden_channels[
                       -1] * nb_time // stride ** nb_layer * nb_freq // stride ** nb_layer

        self.__clf = nn.Sequential(
            nn.Linear(out_size, 2560),
            nn.LeakyReLU(negative_slope=1e-1),
            nn.Linear(2560, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv = self.__conv(x)
        out = out_conv.flatten(1, -1)

        out_clf = self.__clf(out)

        return out_clf


if __name__ == '__main__':
    rand_data = th.rand(1, 8, 1, 2)

    # rs = ResidualTransConv(8, 24, 16, 3, 4, 2)
    """gu = GatedActUnit(
        8, 10, 16, 3, 2
    )

    o = gu(rand_data)

    print(o.size())"""

    gen = STFTGenerator(8, 2)

    print(gen)

    o = gen(rand_data)

    print("A ", o.size())

    disc = STFTDiscriminator(2)

    print(disc)

    o = disc(o)

    print(o.size())
