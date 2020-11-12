import torch as th
import torch.nn as nn

from utils import SAMPLE_RATE, N_FFT, N_SEC


class Generator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.__tr_cnn = nn.Sequential(
            nn.ConvTranspose2d(
                kernel_size=(5, 5),
                in_channels=in_channels,
                out_channels=in_channels // 2,
                stride=2,
                padding=2,
                output_padding=1
            ),
            nn.ELU(),
            nn.ConvTranspose2d(
                kernel_size=(5, 5),
                in_channels=in_channels // 2,
                out_channels=in_channels // 4,
                stride=2,
                padding=2,
                output_padding=1
            ),
            nn.ELU(),
            nn.ConvTranspose2d(
                kernel_size=(4, 4),
                in_channels=in_channels // 4,
                out_channels=in_channels // 8,
                stride=2,
                padding=1
            ),
            nn.ELU(),
            nn.ConvTranspose2d(
                kernel_size=(3, 3),
                in_channels=in_channels // 8,
                out_channels=in_channels // 16,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Tanh()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__tr_cnn(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()

        self.__cnn = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel * 2,
                kernel_size=(3, 3),
                padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.ELU(),
            nn.Conv2d(
                in_channel * 2, int(in_channel * 2 ** 1.5),
                kernel_size=(3, 3),
                padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.ELU(),
            nn.Conv2d(
                int(in_channel * 2 ** 1.5), int(in_channel * 2 ** 2),
                kernel_size=(5, 5),
                padding=(2, 2)),
            nn.MaxPool2d(4, 4),
            nn.ELU()
        )

        height = N_FFT
        width = int(N_SEC * SAMPLE_RATE) // height

        div_factor = 2 * 2 * 4

        self.__lin = nn.Sequential(
            nn.Linear((width // div_factor) ** 2 * (
                    in_channel * 2 ** 2), 2560),
            nn.ELU(),
            nn.Linear(2560, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__cnn(x)
        out = out.flatten(1, -1)
        out = self.__lin(out)
        return out


class Generator2(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()

        self.__lstm = nn.LSTM(
            in_channel, in_channel * 6, batch_first=True
        )

        self.__lin_real = nn.Linear(in_channel * 6, N_FFT)
        self.__lin_imag = nn.Linear(in_channel * 6, N_FFT)

    def forward(self, x_rand: th.Tensor,
                h_first: th.Tensor, c_first: th.Tensor) -> th.Tensor:
        o, _ = self.__lstm(x_rand, (h_first, c_first))
        o = th.relu(o)
        o_real = th.tanh(self.__lin_real(o))
        o_imag = th.tanh(self.__lin_imag(o))
        return th.stack([o_real, o_imag], dim=1)


class Discriminator2(nn.Module):
    def __init__(self, in_channel: int):
        super().__init__()

        self.__cnn = nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel * 2,
                kernel_size=(5, 3),
                padding=(2, 1)),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.ReLU(),
            nn.Conv2d(
                in_channel * 2, int(in_channel * 2 ** 1.5),
                kernel_size=(5, 3),
                padding=(2, 1)),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.ReLU(),
            nn.Conv2d(
                int(in_channel * 2 ** 1.5), int(in_channel * 2 ** 2),
                kernel_size=(5, 5),
                padding=(2, 2)),
            nn.MaxPool2d(4, 4),
            nn.ReLU()
        )

        height = N_FFT
        width = int(N_SEC * SAMPLE_RATE) // height

        div_factor = 2 * 2 * 4

        self.__lin = nn.Sequential(
            nn.Linear((width // div_factor) ** 2 * (
                    in_channel * 2 ** 2), 2560),
            nn.ReLU(),
            nn.Linear(2560, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__cnn(x)
        out = out.flatten(1, -1)
        out = self.__lin(out)
        return out


def discriminator_loss(y_real: th.Tensor, y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_real) + th.log2(1. - y_fake))


def generator_loss(y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_fake))


if __name__ == '__main__':
    gen = Generator(16)
    disc = Discriminator(2)

    data = th.rand(3, 16, 420, 420)

    print(data.size())

    out_gen = gen(data)
    print(out_gen.size())

    out_disc = disc(out_gen)
    print(out_disc.size())
