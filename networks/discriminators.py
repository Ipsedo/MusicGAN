import torch as th
import torch.nn as nn

from utils import N_FFT, N_SEC
from read_audio import to_tensor

import glob


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.__cnn = nn.Sequential(
            nn.Conv2d(
                2, 5,
                kernel_size=(5, 5),
                padding=2),
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
                kernel_size=(7, 7),
                padding=3),
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

        div_factor = 2 * 2 * 2 * 2

        self.__lin = nn.Sequential(
            nn.Linear(
                16 * (width // div_factor) ** 2,
                4608
            ),
            nn.ReLU(),
            nn.Linear(4608, 1),
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


class Discriminator1D(nn.Module):
    def __init__(self):
        super(Discriminator1D, self).__init__()

        vec_size = 256
        vec_nb = 256

        self.__coeff_lin = nn.Linear(vec_size, vec_size // 2)
        self.__magn_lin = nn.Linear(vec_size, vec_size // 2)

        self.__cnn = nn.Sequential(
            nn.Conv1d(vec_size, 200, kernel_size=5, padding=2),
            nn.MaxPool1d(2, 2),
            nn.GELU(),
            nn.Conv1d(200, 150, kernel_size=5, padding=2),
            nn.MaxPool1d(2, 2),
            nn.GELU(),
            nn.Conv1d(150, 100, kernel_size=5, padding=2),
            nn.MaxPool1d(2, 2),
            nn.GELU(),
            nn.Conv1d(100, 64, kernel_size=5, padding=2),
            nn.MaxPool1d(2, 2),
            nn.GELU(),
        )

        div_factor = 2 * 2 * 2 * 2

        self.__clf = nn.Sequential(
            nn.Linear(64 * vec_nb // div_factor, 2048),
            nn.GELU(),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_coef = self.__coeff_lin(x[:, 0, :, :])
        out_magn = self.__magn_lin(x[:, 1, :, :])
        out = th.cat([out_coef, out_magn], dim=-1).permute(0, 2, 1)

        out_cnn = self.__cnn(out)

        out = out_cnn.flatten(1, -1)
        out_lin = self.__clf(out)

        return out_lin


if __name__ == '__main__':
    w_p = "/home/samuel/Documents/MusicGAN/res/rammstein/(1) Mein Herz Brennt.mp3.wav"
    w_p = glob.glob(w_p)

    print(N_SEC)

    out_data = to_tensor(w_p, N_FFT, N_SEC)
    print(out_data.size())

    disc = Discriminator()

    out_disc = disc(out_data)
    print(out_disc.size())
