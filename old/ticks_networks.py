import torch as th
import torch.nn as nn


##############
# Generator
##############

class TransConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,

    ):
        assert kernel_size % 2 == 0, f"kernel_size % 2 must be 0"
        assert stride % 2 == 0, f"stride % 2 must be 0"

        super(TransConv, self).__init__()

        self.__conv_tr = nn.ConvTranspose1d(
            in_channels, out_channels,
            kernel_size, stride,
            padding=(kernel_size - stride) // 2
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv = self.__conv_tr(x)
        return out_conv


# Gate Unit
class GatedActUnit(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            filter_kernel_size: int,
            filter_stride: int,
            gate_kernel_size: int,
            gate_stride: int
    ):
        super(GatedActUnit, self).__init__()

        self.__filter_conv = TransConv(
            input_channels,
            output_channels,
            filter_kernel_size,
            filter_stride
        )

        self.__gate_conv = TransConv(
            input_channels,
            output_channels,
            gate_kernel_size,
            gate_stride
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_f = self.__filter_conv(x)
        out_g = self.__gate_conv(x)

        return th.tanh(out_f) * th.sigmoid(out_g)


# Residual Block
class ResidualTransConv(nn.Module):
    def __init__(
            self,
            input_channel: int,
            hidden_channel: int,
            output_channel: int,
            kernel_size: int,
            stride: int
    ):
        assert kernel_size % 2 == 0, f"kernel_size % 2 must be 0"
        assert stride % 2 == 0, f"stride % 2 must be 0"

        super(ResidualTransConv, self).__init__()

        self.__tr_conv_block = nn.Sequential(
            nn.Conv1d(
                input_channel, hidden_channel,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                hidden_channel, input_channel,
                kernel_size=kernel_size,
                padding=kernel_size // 2 - 1,
                stride=1
            ),
            nn.ReLU()
        )

        self.__tr_conv_down = nn.Sequential(
            nn.ConvTranspose1d(
                input_channel, output_channel,
                kernel_size=kernel_size,
                padding=(kernel_size - stride) // 2,
                stride=stride
            ),
            nn.ReLU(),
            nn.BatchNorm1d(output_channel)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv_tr = self.__tr_conv_block(x)

        out = x + out_conv_tr
        out = self.__tr_conv_down(out)

        return out


class Generator(nn.Module):
    def __init__(
            self,
            rand_channels: int,
            hidden_channels: int,
            res_hidden_channels: int,
            out_channels: int
    ):
        super(Generator, self).__init__()

        nb_layer = 12

        """filter_kernel_size = 26
        filter_stride = 4

        gate_kernel_size = 26
        gate_stride = 4

        self.__gen = nn.Sequential(*[
            GatedActUnit(
                rand_channels if i == 0 else hidden_channels,
                hidden_channels,
                filter_kernel_size,
                filter_stride,
                gate_kernel_size,
                gate_stride
            )
            for i in range(nb_layer)
        ])"""

        self.__gen = nn.Sequential(*[
            ResidualTransConv(
                rand_channels if i == 0 else hidden_channels,
                res_hidden_channels,
                hidden_channels,
                26, 2
            )
            for i in range(nb_layer)
        ])

        self.__out_conv = nn.ConvTranspose1d(
            hidden_channels, out_channels,
            kernel_size=25, stride=1, padding=12
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_tr_conv = self.__gen(x)

        return th.tanh(self.__out_conv(out_tr_conv))


################
# Discriminator
################

# Designed for 16384 ticks aka ~1 sec at 16000Hz

class DiscBlock(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_size: int = 25,
            stride: int = 2
    ):
        super(DiscBlock, self).__init__()

        self.__conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_c = self.__conv(x)

        return th.relu(out_c)


class Discriminator(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super(Discriminator, self).__init__()

        nb_layer = 8

        self.__conv = nn.Sequential(*[
            DiscBlock(
                input_channels if i == 0 else hidden_channels,
                hidden_channels
            )
            for i in range(nb_layer)
        ])

        out_size = 16384 // (2 ** nb_layer) * hidden_channels

        self.__clf = nn.Sequential(
            nn.Linear(out_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv = self.__conv(x)
        out_conv = out_conv.flatten(-2, -1)

        out_clf = self.__clf(out_conv)

        return out_clf


if __name__ == '__main__':
    d = th.randn(16, 8, 2)

    gen = ResidualTransConv(8, 16, 10, 26, 2)

    o = gen(d)

    print(o.size())