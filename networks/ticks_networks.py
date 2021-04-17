import torch as th
import torch.nn as nn


##############
# Generator
##############

class ConvTrBlock(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
    ):
        super(ConvTrBlock, self).__init__()

        kernel_size: int = 25
        stride: int = 4

        self.__conv = nn.ConvTranspose1d(
            input_channels, output_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
            padding=kernel_size // 2,
            output_padding=kernel_size % stride
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__conv(x)


class GatedActUnit(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(GatedActUnit, self).__init__()

        self.__filter_conv = ConvTrBlock(input_channels, output_channels)
        self.__gate_conv = ConvTrBlock(input_channels, output_channels)

    def forward(self, x: th.Tensor):
        out_f = self.__filter_conv(x)
        out_g = self.__gate_conv(x)

        return th.tanh(out_f) * th.sigmoid(out_g)


class Generator(nn.Module):
    def __init__(
            self, rand_channels: int,
            hidden_channels: int, out_channels: int
    ):
        super(Generator, self).__init__()

        nb_layer = 5

        self.__gen = nn.Sequential(*[
            GatedActUnit(
                rand_channels if i == 0 else hidden_channels,
                hidden_channels
            )
            for i in range(nb_layer)
        ])

        self.__out_conv = nn.ConvTranspose1d(
            hidden_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: th.Tensor):
        out_tr_conv = self.__gen(x)

        return th.tanh(self.__out_conv(out_tr_conv))


################
# Discriminator
################

# Designed for 16000 ticks aka 1 sec at 16000Hz

class DiscBlock(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_size: int = 15,
            stride: int = 4
    ):
        super(DiscBlock, self).__init__()

        self.__conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )

        self.__relu = nn.ReLU()

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_c = self.__conv(x)

        return self.__relu(out_c)


class Discriminator(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super(Discriminator, self).__init__()

        nb_layer = 4

        self.__conv = nn.Sequential(*[
            DiscBlock(
                input_channels if i == 0 else hidden_channels,
                hidden_channels
            )
            for i in range(nb_layer)
        ])

        out_size = 16384 // (4 ** nb_layer) * hidden_channels

        self.__clf = nn.Sequential(
            nn.Linear(out_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv =  self.__conv(x)
        out_conv = out_conv.flatten(-2, -1)

        out_clf = self.__clf(out_conv)

        return out_clf
