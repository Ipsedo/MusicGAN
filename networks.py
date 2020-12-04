import torch as th
import torch.nn as nn

from utils import SAMPLE_RATE, N_FFT, N_SEC


class Generator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.__tr_cnn = nn.Sequential(
            nn.ConvTranspose2d(
                kernel_size=(7, 7),
                in_channels=in_channels,
                out_channels=10,
                stride=2,
                padding=3,
                output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                kernel_size=(5, 5),
                in_channels=10,
                out_channels=8,
                stride=2,
                padding=2,
                output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                kernel_size=(4, 4),
                in_channels=8,
                out_channels=5,
                stride=2,
                padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                kernel_size=(3, 3),
                in_channels=5,
                out_channels=4,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                kernel_size=(3, 3),
                in_channels=4,
                out_channels=2,
                stride=1,
                padding=1
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
            nn.GELU(),
            nn.Conv2d(
                in_channel * 2, int(in_channel * 2 ** 1.5),
                kernel_size=(3, 3),
                padding=(1, 1)),
            nn.MaxPool2d(2, 2),
            nn.GELU(),
            nn.Conv2d(
                int(in_channel * 2 ** 1.5), int(in_channel * 2 ** 2),
                kernel_size=(5, 5),
                padding=(2, 2)),
            nn.MaxPool2d(4, 4),
            nn.GELU()
        )

        height = N_FFT
        width = int(N_SEC * SAMPLE_RATE) // height

        div_factor = 2 * 2 * 4

        self.__lin = nn.Sequential(
            nn.Linear((width // div_factor) ** 2 * (
                    in_channel * 2 ** 2), 2560),
            nn.GELU(),
            nn.Linear(2560, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__cnn(x)
        out = out.flatten(1, -1)
        out = self.__lin(out)
        return out


class Generator2(nn.Module):
    def __init__(self, in_channel: int, hidden_size: int):
        super().__init__()

        self.__lstm = nn.LSTM(
            in_channel, hidden_size,
            batch_first=True, num_layers=2
        )

        self.__lin_real = nn.Linear(hidden_size, N_FFT)
        self.__lin_imag = nn.Linear(hidden_size, N_FFT)

    def forward(self, x_rand: th.Tensor,
                h_first: th.Tensor, c_first: th.Tensor) -> th.Tensor:
        o, _ = self.__lstm(x_rand, (h_first, c_first))

        o_real = th.tanh(self.__lin_real(o))
        o_imag = th.tanh(self.__lin_imag(o))

        return th.stack([o_real, o_imag], dim=1)


class Discriminator2(nn.Module):
    def __init__(self):
        super().__init__()

        self.__cnn = nn.Sequential(
            nn.Conv2d(
                2, 5,
                kernel_size=(3, 3),
                padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(
                5, 8,
                kernel_size=(3, 3),
                padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(
                8, 12,
                kernel_size=(5, 5),
                padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(
                12, 16,
                kernel_size=(5, 5),
                padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        height = N_FFT
        width = int(N_SEC * SAMPLE_RATE) // height

        div_factor = 2 * 2 * 2 * 2

        self.__lin = nn.Sequential(
            nn.Linear(
                (height // div_factor) *
                (width // div_factor) *
                16, 5120
            ),
            nn.ReLU(),
            nn.Linear(5120, 1),
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
