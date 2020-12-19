import torch as th
import torch.nn as nn

import glob

from utils import N_FFT, N_SEC
from read_audio import to_tensor


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        vec_size = 256

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

    def forward(self, x: th.Tensor) -> th.Tensor:
        out_coef = self.__coeff_lin(x[:, 0, :, :])
        out_magn = self.__magn_lin(x[:, 1, :, :])
        out = th.cat([out_coef, out_magn], dim=-1).permute(0, 2, 1)

        out_cnn = self.__cnn(out)
        return out_cnn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        in_channel = 64

        self.__tr_cnn = nn.Sequential(
            nn.ConvTranspose1d(
                in_channel, 125,
                kernel_size=7, padding=3, stride=4, output_padding=3
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                125, 175,
                kernel_size=5, padding=2, stride=2, output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                175, 225,
                kernel_size=5, padding=2, stride=2, output_padding=1
            ),
            nn.GELU(),
            nn.ConvTranspose1d(
                225, 300,
                kernel_size=5, padding=2
            ),
            nn.GELU()
        )

        self.__coef_lin = nn.Linear(300, 256)
        self.__magn_lin = nn.Linear(300, 256)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = self.__tr_cnn(x)

        out_coef = self.__coef_lin(out.permute(0, 2, 1))
        out_magn = self.__magn_lin(out.permute(0, 2, 1))

        return th.stack([out_coef, out_magn], dim=1)


if __name__ == '__main__':
    enc = Encoder()
    dec = Decoder()

    w_p = "/home/samuel/Documents/MusicGAN/res/rammstein/(1) Mein Herz Brennt.mp3.wav"
    w_p = glob.glob(w_p)

    print(N_SEC)

    out_data = to_tensor(w_p, N_FFT, N_SEC)
    print(out_data.size())

    out_enc = enc(out_data)
    print(out_enc.size())

    out_dec = dec(out_enc)
    print(out_dec.size())
