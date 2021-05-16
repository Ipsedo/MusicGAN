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
            nn.LeakyReLU(1e-1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_f = self.__filter_conv(x)
        out_g = self.__gate_conv(x)

        out = th.tanh(out_f) * th.sigmoid(out_g)

        out = self.__tr_conv(out)

        return out


class GeneratorBlock(nn.Module):
    def __init__(
            self,
            input_channel: int,
            hidden_channel: int,
            output_channel: int,
            conv_kernel_size: int,
            convtr_kernel_size: int,
            stride: int
    ):
        super(GeneratorBlock, self).__init__()

        negative_slope = 1e-1
        self.__conv_block = nn.Sequential(
            nn.Conv2d(
                input_channel,
                hidden_channel,
                kernel_size=(
                    conv_kernel_size,
                    conv_kernel_size
                ),
                stride=(1, 1),
                padding=(
                    conv_kernel_size // 2,
                    conv_kernel_size // 2
                )
            ),
            nn.LeakyReLU(negative_slope),
            nn.ConvTranspose2d(
                hidden_channel,
                output_channel,
                kernel_size=(
                    convtr_kernel_size,
                    convtr_kernel_size
                ),
                stride=(stride, stride),
                padding=(
                    (convtr_kernel_size - stride) // 2,
                    (convtr_kernel_size - stride) // 2
                )
            ),
            nn.LeakyReLU(negative_slope)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__conv_block(x)


class TransConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int
    ):
        super(TransConvBlock, self).__init__()

        self.__tr_conv_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                stride=(stride, stride),
                kernel_size=(kernel_size, kernel_size),
                padding=(
                    (kernel_size - stride) // 2,
                    (kernel_size - stride) // 2
                )
            ),
            nn.LeakyReLU(1e-1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__tr_conv_block(x)


class STFTGenerator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
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

        channel_list = [
            (rand_channels, 256),
            (256, 192),
            (192, 128),
            (128, 96),
            (96, 64),
            (64, 32),
            (32, 16)
        ]

        self.__gen = nn.Sequential(*[
            TransConvBlock(
                channel_list[i][0],
                channel_list[i][1],
                4, 2
            )
            for i in range(nb_layer)
        ])

        """self.__gen = nn.Sequential(*[
            GatedActUnit(
                channel_list[i][0],
                channel_list[i][1],
                channel_list[i][2],
                kernel_sizes[i][0],
                kernel_sizes[i][1],
                strides[i]
            )
            for i in range(nb_layer)
        ])"""

        """self.__gen = nn.Sequential(*[
            GeneratorBlock(
                channel_list[i][0],
                channel_list[i][1],
                channel_list[i][2],
                5, 6, 4
            )
            for i in range(nb_layer)
        ])"""

        self.__conv_out = nn.Sequential(
            nn.Conv2d(
                channel_list[-1][1],
                out_channel,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
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

        self.__conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                stride=(stride, stride),
                kernel_size=(
                    kernel_size,
                    kernel_size
                ),
                padding=(
                    kernel_size // 2,
                    kernel_size // 2
                )
            ),
            nn.LeakyReLU(1e-1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__conv(x)


class STFTDiscriminator(nn.Module):
    def __init__(
            self,
            in_channels: int
    ):
        super(STFTDiscriminator, self).__init__()

        nb_layer = 7
        stride = 2

        conv_channels = [
            (in_channels, 16),
            (16, 32),
            (32, 64),
            (64, 96),
            (96, 128),
            (128, 192),
            (192, 256)
        ]

        kernel_size = 3

        self.__conv = nn.Sequential(*[
            ConvBlock(
                conv_channels[i][0],
                conv_channels[i][1],
                kernel_size,
                stride
            )
            for i in range(nb_layer)
        ])

        nb_time = 256
        nb_freq = 512

        out_size = conv_channels[-1][1] * \
                   nb_time // stride ** nb_layer * \
                   nb_freq // stride ** nb_layer

        self.__clf = nn.Sequential(
            nn.Linear(out_size, 2560),
            nn.LeakyReLU(1e-1),
            nn.Linear(2560, 1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv = self.__conv(x)
        out = out_conv.flatten(1, -1)

        out_clf = self.__clf(out)

        return out_clf


if __name__ == '__main__':
    rand_data = th.rand(5, 8, 2, 4)

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
