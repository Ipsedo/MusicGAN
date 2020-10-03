import torch as th
import torch.nn as nn

from utils import SAMPLE_RATE, N_FFT, N_SEC


class Generator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.__tr_cnn = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, kernel_size=(3, 3),
                out_channels=int(in_channels / 2), padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=int(in_channels / 2), kernel_size=(3, 3),
                out_channels=int(in_channels / 2 ** 2), padding=1)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__tr_cnn(x)


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
            nn.MaxPool2d(3, 3),
            nn.ReLU(),
            nn.Conv2d(
                int(in_channel * 2 ** 1.5), int(in_channel * 2 ** 2),
                kernel_size=(5, 5),
                padding=(2, 2)
            ),
            nn.MaxPool2d(5, 5),
            nn.ReLU()
        )

        self.__out_size = ((N_SEC * SAMPLE_RATE // N_FFT) // 2 // 3 // 5) ** 2

        self.__lin = nn.Sequential(
            nn.Linear(self.__out_size * (in_channel * 2 ** 2), 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
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
