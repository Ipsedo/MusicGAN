import torch as th
import torch.nn as nn


##############
# Generator
##############

class ConvTrBlock(nn.Module):
    def __init__(
            self, input_channels: int, output_channels: int
    ):
        super(ConvTrBlock, self).__init__()

        self.__conv = nn.ConvTranspose1d(
            input_channels, output_channels,
            kernel_size=2, stride=2
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

        nb_layer = 7

        self.__gen = nn.ModuleList()

        for i in range(nb_layer):
            self.__gen.append(
                GatedActUnit(
                    rand_channels if i == 0 else hidden_channels,
                    hidden_channels
                )
            )

        self.__out_conv = nn.ConvTranspose1d(
            hidden_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: th.Tensor):
        for layer in self.__gen:
            x = layer(x)

        return self.__out_conv(x)


################
# Discriminator
################

# Designed for 16000 ticks aka 1 sec at 16000Hz

class DiscBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super(DiscBlock, self).__init__()

        self.__conv = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.__mp = nn.MaxPool1d(2, 2)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_c = self.__conv(x)
        out_mp = self.__mp(out_c)

        return th.relu(out_mp)


class Discriminator(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super(Discriminator, self).__init__()

        nb_layer = 5

        self.__conv = nn.Sequential(*[
            DiscBlock(
                input_channels if i == 0 else hidden_channels,
                hidden_channels
            )
            for i in range(nb_layer)
        ])

        out_size = 16000 // (2 ** nb_layer) * hidden_channels

        self.__clf = nn.Sequential(
            nn.Linear(out_size, 3072),
            nn.ReLU(),
            nn.Linear(3072, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_conv =  self.__conv(x)
        out_conv = out_conv.flatten(-2, -1)

        out_clf = self.__clf(out_conv)

        return out_clf
