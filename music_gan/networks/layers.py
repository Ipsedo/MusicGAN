import torch as th
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


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


class CausalTimeConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int],
    ):
        super(CausalTimeConv2d, self).__init__()

        dilation = 1

        self.__conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, 0),
            dilation=(dilation, 1)
        )

        self.__padding = 1
        self.__causal_padding = (kernel_size[0] - 1) * dilation

    def forward(self, x: th.Tensor) -> th.Tensor:
        x_padded = F.pad(
            x,
            [
                self.__padding,
                self.__padding,

                self.__causal_padding,
                0,
            ],
            mode="constant",
            value=0.
        )

        return self.__conv(x_padded)

