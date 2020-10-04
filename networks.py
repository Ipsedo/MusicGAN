import torch as th
import torch.nn as nn

from utils import SAMPLE_RATE, N_FFT, N_SEC


class Generator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.__tr_cnn = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                kernel_size=(5, 5),
                out_channels=int(in_channels / 2),
                padding=2),
            nn.CELU(),
            nn.ConvTranspose2d(
                in_channels=int(in_channels / 2),
                kernel_size=(3, 3),
                out_channels=int(in_channels / 2 ** 1.5),
                padding=1),
            nn.CELU(),
            nn.ConvTranspose2d(
                in_channels=int(in_channels / 2 ** 1.5),
                kernel_size=(3, 3),
                out_channels=int(in_channels / 2 ** 2),
                padding=1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__tr_cnn(x)
        return out


class Generator2(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.__cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                kernel_size=(3, 3),
                out_channels=int(in_channels * 2),
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int(in_channels * 2),
                kernel_size=(3, 3),
                out_channels=int(in_channels * 2 ** 1.5),
                padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=int(in_channels * 2 ** 1.5),
                kernel_size=(5, 5),
                out_channels=int(in_channels * 2 ** 2),
                padding=2),
            nn.ReLU()
        )

        self.__lin = nn.Sequential(
            nn.Linear(int(in_channels * 2 ** 2), int(in_channels * 2 ** 2.5)),
            nn.ReLU(),
            nn.Linear(int(in_channels * 2 ** 2.5), 2)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__cnn(x)
        out = out.permute(0, 2, 3, 1)
        out = self.__lin(out)
        out = out.permute(0, 3, 1, 2)
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
            nn.ReLU(),
            nn.Conv2d(
                in_channel * 2, int(in_channel * 2 ** 1.5),
                kernel_size=(3, 3),
                padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(
                int(in_channel * 2 ** 1.5), int(in_channel * 2 ** 2),
                kernel_size=(5, 5),
                padding=(2, 2)
            ),
            nn.MaxPool2d(3, 3),
            nn.ReLU()
        )

        height = N_FFT // 2
        width = N_SEC * SAMPLE_RATE // height

        div_factor = 2 * 2 * 3

        self.__lin = nn.Sequential(
            nn.Linear(((width // div_factor) * (height // div_factor)) * (
                        in_channel * 2 ** 2), 4864),
            nn.ReLU(),
            nn.Linear(4864, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__cnn(x)
        out = out.flatten(1, -1)
        out = self.__lin(out)
        return out


def discriminator_loss(y_real: th.Tensor, y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_real) + th.log2(1. - y_fake), dim=0)


def generator_loss(y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_fake), dim=0)


if __name__ == '__main__':
    gen = Generator(16)
    disc = Discriminator(2)

    data = th.rand(3, 16, 420, 420)

    print(data.size())

    out_gen = gen(data)
    print(out_gen.size())

    out_disc = disc(out_gen)
    print(out_disc.size())
