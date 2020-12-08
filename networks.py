import torch as th
import torch.nn as nn

from utils import SAMPLE_RATE, N_FFT, N_SEC


class Generator(nn.Module):
    def __init__(self, in_channels: int = 10):
        super().__init__()

        self.__tr_cnn = nn.Sequential(
            nn.ConvTranspose2d(
                kernel_size=(7, 7),
                in_channels=in_channels,
                out_channels=8,
                stride=4,
                padding=3,
                output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                kernel_size=(5, 5),
                in_channels=8,
                out_channels=5,
                stride=2,
                padding=2,
                output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                kernel_size=(5, 5),
                in_channels=5,
                out_channels=3,
                stride=2,
                padding=2,
                output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose2d(
                kernel_size=(3, 3),
                in_channels=3,
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
                kernel_size=(5, 5),
                padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(
                8, 11,
                kernel_size=(5, 5),
                padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(
                11, 16,
                kernel_size=(7, 7),
                padding=3,
                stride=2),
            nn.ReLU()
        )

        height = N_FFT // 2
        width = height

        div_factor = 2 * 2 * 2

        self.__lin = nn.Sequential(
            nn.Linear(
                16 * (width // div_factor) ** 2,
                4352
            ),
            nn.ReLU(),
            nn.Linear(4352, 1),
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


class Discriminator3(nn.Module):
    def __init__(self, in_channel: int, hidden_size: int):
        super().__init__()
        self.__lstm_real = nn.LSTM(
            in_channel, hidden_size,
            batch_first=True, num_layers=1
        )
        self.__lstm_imag = nn.LSTM(
            in_channel, hidden_size,
            batch_first=True, num_layers=1
        )

        self.__lin = nn.Linear(hidden_size * 2, 1)

    def forward(
            self, x: th.Tensor,
            h: th.Tensor, c: th.Tensor
    ) -> th.Tensor:
        o_real, _ = self.__lstm_real(x[:, 0, :, :], (h, c))
        o_imag, _ = self.__lstm_imag(x[:, 1, :, :], (h, c))

        o = th.cat([o_real, o_imag], dim=-1)

        o = self.__lin(o)

        o = th.sigmoid(o)

        return o


def discriminator_loss(y_real: th.Tensor, y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_real) + th.log2(1. - y_fake))


def generator_loss(y_fake: th.Tensor) -> th.Tensor:
    return -th.mean(th.log2(y_fake))


if __name__ == '__main__':
    gen = Generator2(128, 64)
    disc = Discriminator3(N_FFT, 100)

    data = th.rand(3, 50, 128)
    h = th.rand(2, 3, 64)
    c = th.rand(2, 3, 64)

    print(data.size(), h.size(), c.size())

    out_gen = gen(data, h, c)
    print(out_gen.size())

    h = th.rand(1, 3, 100)
    c = th.rand(1, 3, 100)

    out_disc = disc(out_gen, h, c)
    print(out_disc.size())
