import torch as th
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.__tr_cnn = nn.Sequential(
            nn.ConvTranspose1d(
                in_channels,
                int(in_channels * 1.2 ** 2),
                kernel_size=3,
                stride=1,
                padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(
                int(in_channels * 1.2 ** 2),
                int(in_channels * 1.2 ** 6),
                kernel_size=3,
                stride=1,
                padding=1),
            nn.CELU(),
            nn.ConvTranspose1d(
                int(in_channels * 1.2 ** 6),
                int(in_channels * 1.2 ** 9),
                kernel_size=3,
                stride=1,
                padding=1),
            nn.CELU()
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__tr_cnn(x)


class FFTMaker(nn.Module):
    def __init__(self, tr_cnn_in_channels: int, n_fft: int):
        super().__init__()

        self.__maker = nn.Sequential(
            nn.Linear(
                int(tr_cnn_in_channels * 1.2 ** 9),
                int(tr_cnn_in_channels * 1.2 ** 10)),
            nn.ReLU(),
            nn.Linear(
                int(tr_cnn_in_channels * 1.2 ** 10),
                n_fft)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.__maker(x)


if __name__ == '__main__':
    gen = Generator(128)
    rl_maker = FFTMaker(128, 525)
    im_maker = FFTMaker(128, 525)

    data = th.rand(1, 128, 84)

    print(data.size())

    out = gen(data)

    print(out.size())

    out = out.permute(0, 2, 1)

    real = rl_maker(out)
    imag = im_maker(out)

    print(real.size())
    print(imag.size())
