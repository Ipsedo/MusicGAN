import torch as th
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super(PixelNorm, self).__init__()

        self.__epsilon = epsilon

    def forward(self, x: th.Tensor) -> th.Tensor:
        norm = th.sqrt(
            x.pow(2.).mean(dim=1, keepdim=True) +
            self.__epsilon
        )

        return x / norm

    def __repr__(self):
        return f"PixelNorm(eps={self.__epsilon})"

    def __str__(self):
        return self.__repr__()


class RandPadding2d(nn.Module):
    def __init__(self, pad: int):
        super(RandPadding2d, self).__init__()

        self.__pad = pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, h, w = x.size()

        r = th.randn(
            b, c, h + self.__pad * 2, w + self.__pad * 2,
            device=x.device
        )

        r[:, :, 1:-1, 1:-1] = x

        return r


class CropLast2d(nn.Module):
    def forward(self, x: th.Tensor) -> th.Tensor:
        return x[:, :, :-1, :-1]
