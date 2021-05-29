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
    ):
        super(GatedActUnit, self).__init__()

        self.__filter_conv = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.__gate_conv = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.__tr_conv = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_channels,
                output_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                output_padding=(1, 1)
            ),
            nn.LeakyReLU(1e-1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_f = self.__filter_conv(x)
        out_g = self.__gate_conv(x)

        out = th.tanh(out_f) * th.sigmoid(out_g)

        out = self.__tr_conv(out)

        return out
