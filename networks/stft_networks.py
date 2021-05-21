import torch as th
import torch.nn as nn
import torch.autograd as th_autograd


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

        channel_list = [
            (rand_channels, 224),
            (224, 192),
            (192, 160),
            (160, 128),
            (128, 96),
            (96, 64),
            (64, 32)
        ]

        self.__gen = nn.Sequential(*[
            TransConvBlock(
                channel_list[i][0],
                channel_list[i][1],
                4, 2
            )
            for i in range(nb_layer)
        ])

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
            nn.LeakyReLU(1e-1)
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
            (in_channels, 32),
            (32, 64),
            (64, 96),
            (96, 128),
            (128, 160),
            (160, 192),
            (192, 224)
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

    def gradient_penalty(
            self, x_real: th.Tensor,
            x_gen: th.Tensor
    ) -> th.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        batch_size = x_real.size()[0]
        eps = th.rand(batch_size, 1, 1, 1, device=device)

        x_interpolated = eps * x_real + (1 - eps) * x_gen
        x_interpolated.requires_grad_(True)

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
