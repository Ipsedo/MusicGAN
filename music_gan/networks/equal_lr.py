from typing import Tuple, Optional, List
from math import sqrt

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class EqualLrConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int],
            padding: Tuple[int, int],
            alpha: float = 2.
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        nn.init.zeros_(self.bias.data)
        nn.init.normal_(self.weight.data)

        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        self.__equal_lr = sqrt(alpha / fan_in)

    def forward(self, x: Tensor) -> Tensor:
        return self._conv_forward(
            x,
            self.weight * self.__equal_lr,
            self.bias
        )


class EqualLrConvTr2d(nn.ConvTranspose2d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Tuple[int, int],
            stride: Tuple[int, int],
            padding: Tuple[int, int],
            output_padding: Tuple[int, int],
            alpha: float = 2.
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding
        )

        nn.init.zeros_(self.bias.data)
        nn.init.normal_(self.weight.data)

        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        self.__equal_lr = sqrt(alpha / fan_in)

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        output_padding = self._output_padding(
            x,
            output_size,
            list(self.stride),
            list(self.padding),
            list(self.kernel_size),
            list(self.dilation)
        )

        return F.conv_transpose2d(
            x,
            self.weight * self.__equal_lr,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation
        )


class EqualLrLinear(nn.Linear):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            alpha: float = 2.
    ) -> None:
        super().__init__(in_features, out_features)

        nn.init.zeros_(self.bias.data)
        nn.init.normal_(self.weight.data)

        fan_in = in_features
        self.__equal_lr = sqrt(alpha / fan_in)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight * self.__equal_lr,
            self.bias
        )
